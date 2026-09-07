"""
Forecast evaluation metrics: MAE, RMSE, MAPE, sMAPE.
Includes per-SKU/store breakdown, volume-segment evaluation,
interval calibration, and multi-horizon evaluation.
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


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean residual — positive = overforecast, negative = underforecast."""
    return float(np.mean(predicted - actual))


def evaluate_all(actual: np.ndarray, predicted: np.ndarray, model_name: str = "") -> dict:
    return {
        "model": model_name,
        "MAE": round(mae(actual, predicted), 4),
        "RMSE": round(rmse(actual, predicted), 4),
        "MAPE": round(mape(actual, predicted), 4),
        "sMAPE": round(smape(actual, predicted), 4),
        "Bias": round(bias(actual, predicted), 4),
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


def evaluate_by_group(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    group_cols: list,
) -> pd.DataFrame:
    """
    Compute MAE, RMSE, MAPE, sMAPE, Bias per group (e.g. store_id, product_id).

    Args:
        df: DataFrame with actual and predicted columns plus group columns.
        actual_col: Column name for actual demand.
        predicted_col: Column name for predicted demand.
        group_cols: List of columns to group by (e.g. ['store_id', 'product_id']).

    Returns:
        DataFrame with one row per group and all metrics.
    """
    rows = []
    for keys, grp in df.groupby(group_cols):
        a = grp[actual_col].values
        p = grp[predicted_col].values
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row.update({
            "MAE": round(mae(a, p), 4),
            "RMSE": round(rmse(a, p), 4),
            "MAPE": round(mape(a, p), 4),
            "sMAPE": round(smape(a, p), 4),
            "Bias": round(bias(a, p), 4),
            "n_rows": len(a),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("RMSE", ascending=False).reset_index(drop=True)


def evaluate_by_volume_segment(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    demand_col: str,
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Bin rows by average demand volume and compute metrics per bin.
    Bins: Low / Medium / High demand.

    Args:
        df: DataFrame with actual, predicted and a demand column for binning.
        actual_col: Column name for actual demand.
        predicted_col: Column name for predicted demand.
        demand_col: Column used to determine demand volume bin.
        n_bins: Number of volume segments (default 3).

    Returns:
        DataFrame with one row per volume segment.
    """
    labels = ["Low", "Medium", "High"][:n_bins]
    df = df.copy()
    df["_volume_segment"] = pd.qcut(df[demand_col], q=n_bins, labels=labels)
    return evaluate_by_group(df, actual_col, predicted_col, ["_volume_segment"])


def calibration_check(
    actual: np.ndarray,
    predicted: np.ndarray,
    residual_std: float,
) -> pd.DataFrame:
    """
    Check what percentage of actual values fall within ±1σ, ±2σ, ±3σ bands.
    Expected coverage: 68%, 95%, 99.7% for a normally distributed forecast error.

    Args:
        actual: Array of actual demand values.
        predicted: Array of point forecast values.
        residual_std: Standard deviation of forecast residuals (e.g. from LSTM).

    Returns:
        DataFrame with sigma levels, expected and actual coverage percentages.
    """
    rows = []
    for sigma in [1, 2, 3]:
        lower = predicted - sigma * residual_std
        upper = predicted + sigma * residual_std
        inside = np.mean((actual >= lower) & (actual <= upper)) * 100
        expected = {1: 68.27, 2: 95.45, 3: 99.73}[sigma]
        rows.append({
            "sigma_band": f"±{sigma}σ",
            "expected_coverage_%": expected,
            "actual_coverage_%": round(inside, 2),
            "gap_%": round(inside - expected, 2),
        })
    return pd.DataFrame(rows)


def evaluate_multi_horizon(
    df: pd.DataFrame,
    actual_col: str,
    horizon_cols: dict,
) -> pd.DataFrame:
    """
    Evaluate forecast accuracy at multiple horizons.

    Args:
        df: DataFrame with actual and horizon prediction columns.
        actual_col: Column name for actual demand.
        horizon_cols: Dict mapping horizon label to column name,
                      e.g. {'1-day': 'pred_h1', '7-day': 'pred_h7', '30-day': 'pred_h30'}.

    Returns:
        DataFrame with one row per horizon showing all metrics.
    """
    rows = []
    a = df[actual_col].values
    for horizon_label, pred_col in horizon_cols.items():
        p = df[pred_col].values
        rows.append({
            "horizon": horizon_label,
            "MAE": round(mae(a, p), 4),
            "RMSE": round(rmse(a, p), 4),
            "MAPE": round(mape(a, p), 4),
            "sMAPE": round(smape(a, p), 4),
            "Bias": round(bias(a, p), 4),
        })
    return pd.DataFrame(rows).set_index("horizon")
