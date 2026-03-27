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
