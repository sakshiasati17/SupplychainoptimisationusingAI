"""
Demand forecasting models:
- Naive Baseline (last value)
- Moving Average
- ARIMA / SARIMA
- Prophet
- XGBoost Regression
- LightGBM Regression
- LSTM (PyTorch)
"""
import warnings
import numpy as np
import pandas as pd
from typing import Tuple

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Naive & Moving Average Baselines
# ---------------------------------------------------------------------------

def naive_forecast(series: pd.Series) -> pd.Series:
    """Predict t+1 = t (last observed value)."""
    return series.shift(1)


def moving_average_forecast(series: pd.Series, window: int = 7) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# ARIMA
# ---------------------------------------------------------------------------

def train_arima(train: pd.Series, order: Tuple = (1, 1, 1)):
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(train, order=order)
    return model.fit()


def forecast_arima(fitted_model, steps: int = 30) -> pd.Series:
    forecast = fitted_model.forecast(steps=steps)
    return forecast


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

def train_sarima(train: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    return model.fit(disp=False)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def train_prophet(df: pd.DataFrame, date_col: str = "Date", target_col: str = "Units_Sold"):
    from prophet import Prophet
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(prophet_df)
    return model


def forecast_prophet(model, periods: int = 30, freq: str = "D") -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
    from xgboost import XGBRegressor
    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)
    model = XGBRegressor(**default_params)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, params: dict = None):
    from lightgbm import LGBMRegressor
    default_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        default_params.update(params)
    model = LGBMRegressor(**default_params)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------

def create_sequences(data: np.ndarray, seq_len: int = 14):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


class LSTMForecaster:
    def __init__(self, seq_len: int = 14, hidden_size: int = 64, epochs: int = 20):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.model = None
        self.scaler = None

    def fit(self, series: pd.Series):
        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler

        self.scaler = MinMaxScaler()
        values = self.scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        X, y = create_sequences(values, self.seq_len)

        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y)

        class _LSTM(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze()

        self.model = _LSTM(self.hidden_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            pred = self.model(X_tensor)
            loss = loss_fn(pred, y_tensor)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, series: pd.Series, steps: int = 30) -> np.ndarray:
        import torch
        self.model.eval()
        values = self.scaler.transform(series.values[-self.seq_len :].reshape(-1, 1)).flatten()
        preds = []
        with torch.no_grad():
            seq = list(values)
            for _ in range(steps):
                x = torch.FloatTensor(seq[-self.seq_len :]).unsqueeze(0).unsqueeze(-1)
                p = self.model(x).item()
                preds.append(p)
                seq.append(p)
        return self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
