"""
Data preprocessing pipeline for retail inventory demand forecasting.
"""
import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/retail_store_inventory.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_retail_data.csv")


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=["Units_Sold", "Inventory_Level", "Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store_ID", "Product_ID", "Date"]).reset_index(drop=True)

    # Outlier removal via IQR on Units_Sold
    Q1 = df["Units_Sold"].quantile(0.25)
    Q3 = df["Units_Sold"].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df["Units_Sold"] >= Q1 - 1.5 * IQR) & (df["Units_Sold"] <= Q3 + 1.5 * IQR)]

    # Fill remaining numeric nulls with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily store-product level."""
    agg = (
        df.groupby(["Date", "Store_ID", "Product_ID", "Category", "Region"])
        .agg(
            Units_Sold=("Units_Sold", "sum"),
            Inventory_Level=("Inventory_Level", "mean"),
            Units_Ordered=("Units_Ordered", "sum"),
            Price=("Price", "mean"),
            Discount=("Discount", "mean"),
        )
        .reset_index()
    )
    return agg


def run_pipeline() -> pd.DataFrame:
    df = load_raw_data()
    df = clean_data(df)
    df = aggregate_daily(df)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Saved cleaned data → {PROCESSED_DATA_PATH} ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    run_pipeline()
