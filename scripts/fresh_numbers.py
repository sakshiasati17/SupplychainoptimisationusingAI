"""
Fresh, leakage-free numbers for the project.

Run from the repo root:  python scripts/fresh_numbers.py

Produces (printed and saved to reports/fresh_numbers/):
  1. Global XGBoost on a date-based 30-day holdout, with and without the
     dataset's own Demand_Forecast column (no Inventory_Turnover leakage)
  2. LSTM vs Naive / Moving Average / ARIMA on the same single series as NB03
  3. Quantile XGBoost (q10/q50/q90) coverage and pinball loss on real data
  4. Rolling-origin backtest on real data (3 x ~30-day folds)
  5. Inventory simulation for all 100 store-product pairs:
     original simulator vs fixed simulator vs forecast-driven policies
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.models.evaluation import evaluate_all, compare_models, rolling_origin_backtest
from src.models.quantile_forecast import train_quantile_models, evaluate_quantile_models, predict_intervals
from src.simulation.inventory import InventoryPolicy, simulate_inventory, compute_eoq

OUT = Path("reports/fresh_numbers")
OUT.mkdir(parents=True, exist_ok=True)
TARGET = "Units_Sold"
TEST_DAYS = 30


def xgb_model():
    from xgboost import XGBRegressor
    return XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Data + feature sets
# ---------------------------------------------------------------------------
feat = pd.read_csv("data/processed/feature_matrix.csv", parse_dates=["Date"])
feat = feat.sort_values(["Date", "Store_ID", "Product_ID"]).reset_index(drop=True)

PAST = [c for c in feat.columns if "_lag_" in c or "_rolling_" in c]
CALENDAR = ["day_of_week", "day_of_month", "month", "quarter", "year", "week_of_year",
            "is_weekend", "month_sin", "month_cos", "dow_sin", "dow_cos"]
KNOWN_AHEAD = ["Price", "Discount", "Holiday_Promotion", "Competitor_Pricing"]
OPTION_B = PAST + CALENDAR + KNOWN_AHEAD          # past sales + plan-known inputs
OPTION_A = OPTION_B + ["Demand_Forecast"]          # + the dataset's own forecast
# Deliberately excluded (same-day leakage): Inventory_Turnover, Inventory_Level,
# Units_Ordered, demand_inventory_gap, stockout_flag, overstock_flag

cutoff = feat["Date"].max() - pd.Timedelta(days=TEST_DAYS - 1)
train, test = feat[feat["Date"] < cutoff], feat[feat["Date"] >= cutoff]
y_tr, y_te = train[TARGET].values, test[TARGET].values

# ---------------------------------------------------------------------------
# 1. Global model, date-based split
# ---------------------------------------------------------------------------
section(f"1. GLOBAL MODELS — last {TEST_DAYS} days ({test['Date'].min().date()} → "
        f"{test['Date'].max().date()}), {len(train):,} train / {len(test):,} test rows")
results = [
    evaluate_all(y_te, test["Units_Sold_lag_1"].values, "Naive (yesterday)"),
    evaluate_all(y_te, test["Units_Sold_rolling_mean_7"].values, "Moving avg 7d"),
    evaluate_all(y_te, test["Units_Sold_rolling_mean_28"].values, "Moving avg 28d"),
    evaluate_all(y_te, test["Demand_Forecast"].values, "Dataset Demand_Forecast as-is"),
]
for label, cols in [("XGBoost B (no Demand_Forecast)", OPTION_B),
                    ("XGBoost A (+ Demand_Forecast)", OPTION_A)]:
    m = xgb_model().fit(train[cols], y_tr)
    results.append(evaluate_all(y_te, m.predict(test[cols]), label))
    if label.startswith("XGBoost A"):
        top = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False).head(5)
        print("Top features (A):", ", ".join(f"{k}={v:.2f}" for k, v in top.items()))
global_table = compare_models(results)
print(global_table.to_string())
global_table.to_csv(OUT / "1_global_models.csv")

# ---------------------------------------------------------------------------
# 2. Single series: LSTM vs baselines vs ARIMA (same setup as NB03)
# ---------------------------------------------------------------------------
section("2. SINGLE SERIES S001 / P0001 — 30-day ahead forecast")
clean = pd.read_csv("data/processed/cleaned_retail_data.csv", parse_dates=["Date"])
clean.columns = clean.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
ts = (clean[(clean["Store_ID"] == "S001") & (clean["Product_ID"] == "P0001")]
      .groupby("Date")[TARGET].sum().sort_index())
s_tr, s_te = ts.iloc[:-TEST_DAYS], ts.iloc[-TEST_DAYS:].values
single = [
    evaluate_all(s_te, np.full(TEST_DAYS, s_tr.iloc[-1]), "Naive"),
    evaluate_all(s_te, np.full(TEST_DAYS, s_tr.rolling(7).mean().iloc[-1]), "Moving avg 7d"),
]
try:
    from src.models.forecasting import train_arima
    single.append(evaluate_all(s_te, train_arima(s_tr).forecast(TEST_DAYS).values, "ARIMA(1,1,1)"))
except Exception as e:
    print("ARIMA skipped:", e)
try:
    import torch
    from src.models.forecasting import LSTMForecaster
    for ep, lbl in [(20, "LSTM (20 epochs)"), (200, "LSTM (200 epochs)")]:
        torch.manual_seed(42)
        lstm = LSTMForecaster(seq_len=14, hidden_size=64, epochs=ep).fit(s_tr)
        single.append(evaluate_all(s_te, lstm.predict(s_tr, steps=TEST_DAYS), lbl))
except Exception as e:
    print("LSTM skipped:", e)
single_table = compare_models(single)
print(single_table.to_string())
single_table.to_csv(OUT / "2_single_series.csv")

# ---------------------------------------------------------------------------
# 3. Quantile XGBoost on real data
# ---------------------------------------------------------------------------
section("3. QUANTILE XGBOOST (q10 / q50 / q90) — same 30-day holdout")
for label, cols in [("B", OPTION_B), ("A", OPTION_A)]:
    qm = train_quantile_models(train[cols].values, y_tr)
    qtab = evaluate_quantile_models(qm, test[cols].values, y_te)
    iv = predict_intervals(qm, test[cols].values)
    inside = np.mean((y_te >= iv["q10"].values) & (y_te <= iv["q90"].values)) * 100
    print(f"\nOption {label}: 80% interval coverage = {inside:.1f}% (target 80%), "
          f"mean width = {iv['interval_width'].mean():.1f} units")
    print(qtab.to_string())
    qtab.assign(interval_coverage_80=round(inside, 2),
                mean_width=round(iv["interval_width"].mean(), 2)).to_csv(OUT / f"3_quantile_{label}.csv")

# ---------------------------------------------------------------------------
# 4. Rolling-origin backtest on real data
# ---------------------------------------------------------------------------
section("4. ROLLING-ORIGIN BACKTEST — 3 expanding folds of ~30 days")
rows_per_30_days = int(feat.groupby("Date").size().mean() * TEST_DAYS)
for label, cols in [("B", OPTION_B), ("A", OPTION_A)]:
    bt = rolling_origin_backtest(feat, cols, TARGET, xgb_model, n_splits=3,
                                 test_size=rows_per_30_days, min_train_size=10_000)
    print(f"\nOption {label}:")
    print(bt[["MAE", "RMSE", "sMAPE", "Bias"]].to_string())
    bt.to_csv(OUT / f"4_backtest_{label}.csv")

# ---------------------------------------------------------------------------
# 5. Inventory simulation — all 100 store-product pairs, last 180 days
# ---------------------------------------------------------------------------
SIM_DAYS, L, Z, H, S = 180, 3, 1.65, 0.5, 5.0


def simulate_position(actual, lt_forecast, ss, Q, init):
    """Reorder when inventory position (on hand + on order) <= lead-time forecast + SS."""
    inv, pipeline = init, {}
    held = lost_cost = 0.0
    stock_days = over_days = orders = 0
    for t, d in enumerate(actual):
        inv += pipeline.pop(t, 0.0)
        sold = min(d, inv)
        lost_cost += (d - sold) * S
        inv -= sold
        stock_days += inv <= 0
        over_days += inv > 2 * d + 1
        held += inv * H
        if inv + sum(pipeline.values()) <= lt_forecast[t] + ss:
            pipeline[t + L] = pipeline.get(t + L, 0.0) + Q
            orders += 1
    n = len(actual)
    return {"service_level_%": 100 - 100 * stock_days / n, "stockout_rate_%": 100 * stock_days / n,
            "overstock_rate_%": 100 * over_days / n, "avg_inventory": held / H / n,
            "holding_cost_$": held, "lost_sales_cost_$": lost_cost,
            "total_cost_$": held + lost_cost, "orders": orders, "safety_stock": ss}


def lead_time_actual(x):
    """Actual demand over days t+1..t+L, aligned to day t."""
    return pd.Series(x).rolling(L).sum().shift(-L).values


section(f"5. INVENTORY SIMULATION — 100 pairs, last {SIM_DAYS} days, L={L}, z={Z}")
daily = (feat.groupby(["Store_ID", "Product_ID", "Date"])[[TARGET, "Demand_Forecast"]]
         .mean().reset_index().sort_values("Date"))
sim_rows = []
for (store, product), g in daily.groupby(["Store_ID", "Product_ID"]):
    d = g[TARGET].values
    fc = g["Demand_Forecast"].clip(lower=0).values
    hist, act = slice(0, len(d) - SIM_DAYS), slice(len(d) - SIM_DAYS, len(d))
    mu, sigma = d[hist].mean(), d[hist].std()
    Q = compute_eoq(mu * 365)
    ss_static = Z * sigma * np.sqrt(L)
    init = mu * L + ss_static

    # Lead-time demand forecasts known at the end of day t
    lt_true = lead_time_actual(d)
    lt_ma = pd.Series(d).rolling(28, min_periods=7).mean().values * L
    lt_df = lead_time_actual(fc)  # assumes Demand_Forecast is published >= L days ahead

    def ss_from_error(lt_fc):
        err = (lt_true - lt_fc)[hist]
        return Z * np.nanstd(err)

    policies = {
        "0 Original simulator logic (as used for README 99.77%)": None,
        "1 Fixed simulator, static ROP, demand-sigma SS": (np.full(len(d), mu * L), ss_static),
        "2 Moving-avg forecast, error-sigma SS": (lt_ma, ss_from_error(lt_ma)),
        "3 Demand_Forecast, error-sigma SS": (lt_df, ss_from_error(lt_df)),
    }
    for name, spec in policies.items():
        if spec is None:
            res = simulate_inventory(d[act], InventoryPolicy(mu * L + ss_static, Q, L, H, S, init), track_pipeline=False).summary()
            res = {"service_level_%": res["service_level_%"], "stockout_rate_%": res["stockout_rate_%"],
                   "overstock_rate_%": res["overstock_rate_%"],
                   "avg_inventory": res["total_holding_cost_$"] / H / SIM_DAYS,
                   "holding_cost_$": res["total_holding_cost_$"],
                   "lost_sales_cost_$": res["total_lost_sales_cost_$"],
                   "total_cost_$": res["total_operating_cost_$"],
                   "orders": res["n_orders_placed"], "safety_stock": ss_static}
        else:
            lt_fc, ss = spec
            lt_fc = np.nan_to_num(lt_fc, nan=mu * L)
            res = simulate_position(d[act], lt_fc[act], ss, Q, init)
        sim_rows.append({"Store": store, "Product": product, "policy": name, **res})

sim = pd.DataFrame(sim_rows)
sim.to_csv(OUT / "5_simulation_by_pair.csv", index=False)
agg = sim.groupby("policy").agg(
    service_level=("service_level_%", "mean"), stockout_rate=("stockout_rate_%", "mean"),
    overstock_rate=("overstock_rate_%", "mean"), avg_inventory=("avg_inventory", "mean"),
    avg_safety_stock=("safety_stock", "mean"), total_cost=("total_cost_$", "sum"),
).round(2)
base_inv = agg.loc["1 Fixed simulator, static ROP, demand-sigma SS", "avg_inventory"]
agg["inventory_vs_policy1_%"] = ((agg["avg_inventory"] / base_inv - 1) * 100).round(1)
print(agg.to_string())
agg.to_csv(OUT / "5_simulation_summary.csv")
print(f"\nAll outputs saved to {OUT}/")
