"""
Forecast evaluation metrics: MAE, RMSE, MAPE, sMAPE.
"""
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(actual - predicted))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    return np.mean(np.abs((actual - predicted) / (np.abs(actual) + eps))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-8) -> float:
    return (
        np.mean(
            2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + eps)
        )
        * 100
    )


def evaluate_all(actual: np.ndarray, predicted: np.ndarray, model_name: str = "") -> dict:
    return {
        "model": model_name,
        "MAE": round(mae(actual, predicted), 4),
        "RMSE": round(rmse(actual, predicted), 4),
        "MAPE": round(mape(actual, predicted), 4),
        "sMAPE": round(smape(actual, predicted), 4),
    }


def compare_models(results: list) -> pd.DataFrame:
    """
    results: list of dicts from evaluate_all()
    Returns a DataFrame sorted by RMSE ascending.
    """
    df = pd.DataFrame(results)
    if "model" in df.columns:
        df = df.set_index("model")
    return df.sort_values("RMSE")


def rolling_origin_backtest(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model_fn,
    n_splits: int = 5,
    test_size: int = 30,
    min_train_size: int = 90,
) -> pd.DataFrame:
    """
    Rolling-origin (expanding window) cross-validation for time series.

    Unlike k-fold, each split trains on all data up to a cutoff point and
    tests on the next test_size days. The training window expands with
    each split — no future data ever leaks into training.

    Split layout (n_splits=3, test_size=30):
      Split 1: train=[0..T-90], test=[T-90..T-60]
      Split 2: train=[0..T-60], test=[T-60..T-30]
      Split 3: train=[0..T-30], test=[T-30..T]

    Args:
        df: Feature matrix sorted chronologically.
        feature_cols: Input feature column names.
        target_col: Target demand column name.
        model_fn: Callable returning a fresh unfitted sklearn-compatible model.
        n_splits: Number of train/test folds.
        test_size: Number of rows per test fold.
        min_train_size: Minimum rows required in the training fold.

    Returns:
        DataFrame with one row per split: MAE, RMSE, MAPE, sMAPE, Bias,
        plus train/test sizes and cutoff index.
    """
    n = len(df)
    required = min_train_size + n_splits * test_size
    if n < required:
        raise ValueError(
            f"Not enough rows ({n}) for {n_splits} splits with "
            f"test_size={test_size} and min_train_size={min_train_size}. "
            f"Need at least {required} rows."
        )

    X = df[feature_cols].values
    y = df[target_col].values

    results = []
    for i in range(n_splits):
        test_end = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size
        train_end = test_start

        if train_end < min_train_size:
            continue

        X_train, y_train = X[:train_end], y[:train_end]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]

        model = model_fn()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "split": i + 1,
            "train_size": train_end,
            "test_size": len(y_test),
            "cutoff_idx": test_start,
            "MAE": round(mae(y_test, preds), 4),
            "RMSE": round(rmse(y_test, preds), 4),
            "MAPE": round(mape(y_test, preds), 4),
            "sMAPE": round(smape(y_test, preds), 4),
            "Bias": round(float(np.mean(preds - y_test)), 4),
        })

    summary = pd.DataFrame(results).set_index("split")
    agg = summary[["MAE", "RMSE", "MAPE", "sMAPE", "Bias"]].agg(["mean", "std"]).round(4)
    agg.index = [f"mean_across_splits", f"std_across_splits"]
    return pd.concat([summary, agg])
