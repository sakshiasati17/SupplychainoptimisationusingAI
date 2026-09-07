"""
Multi-horizon forecast target generation.

Generates shifted target columns for 1-day, 7-day, and 30-day horizons
from an existing feature matrix so the same model architecture can be
trained and evaluated at each horizon independently.
"""
import numpy as np
import pandas as pd


HORIZONS = {
    "1-day": 1,
    "7-day": 7,
    "30-day": 30,
}


def add_horizon_targets(
    df: pd.DataFrame,
    demand_col: str = "units_sold",
    group_cols: list = None,
    horizons: dict = None,
) -> pd.DataFrame:
    """
    Add forward-shifted target columns for multiple forecast horizons.

    For each horizon h, the target at row t is demand at t+h.
    Rows where the shifted target is NaN (end of each series) are dropped.

    Args:
        df: Feature matrix with a demand column and optional group columns.
        demand_col: Column containing the demand values to shift.
        group_cols: Columns identifying each time series (e.g. ['store_id', 'product_id']).
                    If None, shifts are applied globally.
        horizons: Dict of {label: shift_days}. Defaults to 1, 7, 30 day horizons.

    Returns:
        DataFrame with additional columns 'target_h1', 'target_h7', 'target_h30'
        (or matching the provided horizon labels).
    """
    if horizons is None:
        horizons = HORIZONS

    df = df.copy()

    for label, shift in horizons.items():
        col_name = f"target_h{shift}"
        if group_cols:
            df[col_name] = (
                df.groupby(group_cols)[demand_col]
                .shift(-shift)
            )
        else:
            df[col_name] = df[demand_col].shift(-shift)

    target_cols = [f"target_h{s}" for s in horizons.values()]
    df = df.dropna(subset=target_cols).reset_index(drop=True)
    return df


def train_horizon_models(
    df: pd.DataFrame,
    feature_cols: list,
    horizons: dict = None,
    test_days: int = 30,
    model_fn=None,
):
    """
    Train and evaluate a model at each forecast horizon using chronological split.

    Args:
        df: Feature matrix with horizon target columns added via add_horizon_targets().
        feature_cols: List of input feature column names.
        horizons: Dict of {label: shift_days}. Defaults to 1, 7, 30 day horizons.
        test_days: Number of days to hold out for evaluation.
        model_fn: Callable that returns a fitted sklearn-compatible model.
                  Defaults to XGBoost regressor if None.

    Returns:
        Dict of {horizon_label: {'model': fitted_model, 'mae': float, 'rmse': float}}.
    """
    import importlib
    from src.models.evaluation import mae, rmse, mape, smape, bias

    if horizons is None:
        horizons = HORIZONS

    if model_fn is None:
        xgb = importlib.import_module("xgboost")

        def model_fn():
            return xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )

    results = {}
    cutoff = df.index[-test_days] if len(df) > test_days else df.index[len(df) // 2]

    for label, shift in horizons.items():
        target_col = f"target_h{shift}"
        if target_col not in df.columns:
            raise ValueError(
                f"Column '{target_col}' not found. Run add_horizon_targets() first."
            )

        train = df[df.index < cutoff]
        test = df[df.index >= cutoff]

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[label] = {
            "model": model,
            "mae": round(mae(y_test, preds), 4),
            "rmse": round(rmse(y_test, preds), 4),
            "mape": round(mape(y_test, preds), 4),
            "smape": round(smape(y_test, preds), 4),
            "bias": round(bias(y_test, preds), 4),
            "n_test": len(y_test),
        }

    return results


def horizon_results_table(results: dict) -> pd.DataFrame:
    """Convert train_horizon_models output to a clean comparison DataFrame."""
    rows = []
    for label, metrics in results.items():
        row = {"horizon": label}
        row.update({k: v for k, v in metrics.items() if k != "model"})
        rows.append(row)
    return pd.DataFrame(rows).set_index("horizon")
