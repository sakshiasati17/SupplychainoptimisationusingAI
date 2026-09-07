"""
Probabilistic demand forecasting via quantile regression.

Trains three XGBoost models at the 10th, 50th, and 90th percentiles
using pinball (quantile) loss. Produces calibrated prediction intervals
that are directly usable as safety stock inputs.

Pinball loss for quantile q:
    L(y, f) = q * (y - f)       if y >= f   (underforecast penalty)
             (1-q) * (f - y)    if y <  f   (overforecast penalty)

A 90th-percentile forecast means: 90% of actual demand falls below this value.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional


QUANTILES = {
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
}


def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    quantiles: dict = None,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    random_state: int = 42,
) -> dict:
    """
    Train one XGBoost quantile regression model per quantile.

    Args:
        X_train: Feature matrix (n_samples, n_features).
        y_train: Target demand values.
        quantiles: Dict of {label: alpha} — defaults to 10th, 50th, 90th.
        n_estimators: Number of boosting rounds.
        learning_rate: Step size shrinkage.
        max_depth: Maximum tree depth.
        random_state: Reproducibility seed.

    Returns:
        Dict of {label: fitted XGBRegressor}.
    """
    import xgboost as xgb

    if quantiles is None:
        quantiles = QUANTILES

    models = {}
    for label, alpha in quantiles.items():
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            random_state=random_state,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        models[label] = model

    return models


def predict_intervals(
    models: dict,
    X: np.ndarray,
) -> pd.DataFrame:
    """
    Generate prediction intervals from fitted quantile models.

    Args:
        models: Dict of {label: fitted model} from train_quantile_models().
        X: Feature matrix for prediction.

    Returns:
        DataFrame with columns ['q10', 'q50', 'q90'] and interval width.
    """
    preds = {}
    for label, model in models.items():
        preds[label] = model.predict(X)

    df = pd.DataFrame(preds)
    df["interval_width"] = df.get("q90", 0) - df.get("q10", 0)
    return df


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, alpha: float) -> float:
    """
    Compute mean pinball (quantile) loss.

    Lower is better. At alpha=0.5 this equals half the MAE.

    Args:
        actual: True demand values.
        predicted: Quantile forecast values.
        alpha: Target quantile (0 < alpha < 1).

    Returns:
        Mean pinball loss.
    """
    errors = actual - predicted
    loss = np.where(errors >= 0, alpha * errors, (alpha - 1) * errors)
    return float(np.mean(loss))


def evaluate_quantile_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    quantiles: dict = None,
) -> pd.DataFrame:
    """
    Evaluate quantile models on test set using pinball loss and coverage.

    Coverage = fraction of actual values that fall below the predicted quantile.
    A well-calibrated q90 model should have ~90% coverage.

    Args:
        models: Dict of {label: fitted model}.
        X_test: Test feature matrix.
        y_test: True demand values.
        quantiles: Dict of {label: alpha} matching the models dict.

    Returns:
        DataFrame with pinball loss and coverage per quantile.
    """
    if quantiles is None:
        quantiles = QUANTILES

    interval_df = predict_intervals(models, X_test)
    rows = []
    for label, alpha in quantiles.items():
        if label not in interval_df.columns:
            continue
        preds = interval_df[label].values
        pb = pinball_loss(y_test, preds, alpha)
        coverage = float(np.mean(y_test <= preds) * 100)
        rows.append({
            "quantile": label,
            "alpha": alpha,
            "target_coverage_%": round(alpha * 100, 1),
            "actual_coverage_%": round(coverage, 2),
            "calibration_gap_%": round(coverage - alpha * 100, 2),
            "pinball_loss": round(pb, 4),
        })

    return pd.DataFrame(rows).set_index("quantile")


def safety_stock_from_quantile(
    q90_forecast: np.ndarray,
    q50_forecast: np.ndarray,
    lead_time: int = 7,
) -> float:
    """
    Derive safety stock from quantile interval instead of residual std.

    The gap between q90 and q50 represents the upside uncertainty.
    Summing over lead_time captures cumulative uncertainty during replenishment.

    Args:
        q90_forecast: 90th percentile daily demand forecasts.
        q50_forecast: Median (50th percentile) daily demand forecasts.
        lead_time: Replenishment lead time in days.

    Returns:
        Safety stock in units.
    """
    daily_buffer = np.mean(q90_forecast - q50_forecast)
    return round(float(daily_buffer * np.sqrt(lead_time)), 2)


def save_quantile_models(models: dict, output_dir: str = "models") -> None:
    """Save each quantile model to disk."""
    Path(output_dir).mkdir(exist_ok=True)
    for label, model in models.items():
        path = Path(output_dir) / f"xgboost_{label}.pkl"
        joblib.dump(model, path)


def load_quantile_models(output_dir: str = "models", quantiles: dict = None) -> dict:
    """Load previously saved quantile models from disk."""
    if quantiles is None:
        quantiles = QUANTILES
    models = {}
    for label in quantiles:
        path = Path(output_dir) / f"xgboost_{label}.pkl"
        if path.exists():
            models[label] = joblib.load(path)
    return models
