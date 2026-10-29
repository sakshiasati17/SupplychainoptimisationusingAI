"""Retrain the dashboard's XGBoost/LightGBM without leaky features.
Run from the repo root: python scripts/retrain_dashboard_models.py"""
import joblib
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

feat = pd.read_csv("data/processed/feature_matrix.csv", parse_dates=["Date"]).sort_values("Date")
PAST = [c for c in feat.columns if "_lag_" in c or "_rolling_" in c]
CAL = ["day_of_week", "day_of_month", "month", "quarter", "year", "week_of_year",
       "is_weekend", "month_sin", "month_cos", "dow_sin", "dow_cos"]
KNOWN = ["Price", "Discount", "Holiday_Promotion", "Competitor_Pricing"]
cols = PAST + CAL + KNOWN  # no Demand_Forecast, no Inventory_Turnover

cutoff = feat["Date"].max() - pd.Timedelta(days=29)  # last 30 days stay unseen
train = feat[feat["Date"] < cutoff]
xgb = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                   colsample_bytree=0.8, random_state=42, n_jobs=1).fit(train[cols], train["Units_Sold"])
lgb = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, num_leaves=31,
                    subsample=0.8, random_state=42, n_jobs=1, verbose=-1).fit(train[cols], train["Units_Sold"])

joblib.dump(xgb, "models/xgboost_demand.pkl")
joblib.dump(lgb, "models/lightgbm_demand.pkl")
joblib.dump(cols, "models/feature_cols.pkl")
print(f"Saved models: {len(cols)} features, trained on data before {cutoff.date()}")
