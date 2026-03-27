"""
Feature engineering for demand forecasting.
Creates time-based, lag, and rolling features from cleaned retail data.
"""
import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["day_of_month"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["quarter"] = df[date_col].dt.quarter
    df["year"] = df[date_col].dt.year
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding to preserve periodicity
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "Units_Sold",
    group_cols: list = ["Store_ID", "Product_ID"],
    lags: list = [1, 7, 14, 28],
) -> pd.DataFrame:
    df = df.copy().sort_values(group_cols + ["Date"])
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby(group_cols)[target_col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "Units_Sold",
    group_cols: list = ["Store_ID", "Product_ID"],
    windows: list = [7, 14, 28],
) -> pd.DataFrame:
    df = df.copy().sort_values(group_cols + ["Date"])
    for w in windows:
        rolled = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{target_col}_rolling_mean_{w}"] = rolled
        rolled_std = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(w, min_periods=1).std()
        )
        df[f"{target_col}_rolling_std_{w}"] = rolled_std
    return df


def add_inventory_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Inventory_Turnover"] = df["Units_Sold"] / (df["Inventory_Level"] + 1)
    df["demand_inventory_gap"] = df["Units_Sold"] - df["Inventory_Level"]
    df["stockout_flag"] = (df["Units_Sold"] > df["Inventory_Level"]).astype(int)
    df["overstock_flag"] = (df["Inventory_Level"] > 2 * df["Units_Sold"]).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_inventory_features(df)
    df = df.dropna()
    return df
