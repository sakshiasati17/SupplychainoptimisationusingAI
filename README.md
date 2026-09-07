# Supply Chain Demand Forecasting & Inventory Optimization

**End-to-end retail inventory decision support system — from raw sales data to probabilistic forecasts, inventory simulation, REST API, and interactive dashboard.**

---

## Verified Numbers (traceable from code and outputs)

| Fact | Value | Source |
|---|---|---|
| Raw records | 73,185 | `data/raw/retail_store_inventory.csv` |
| Cleaned records | 72,287 | `data/processed/cleaned_retail_data.csv` |
| Feature matrix rows | 70,209 | `data/processed/feature_matrix.csv` |
| Model features | 24 | `models/feature_cols.pkl` |
| Stores | 5 (S001–S005) | raw data |
| Products | 21 | raw data |
| Categories | 5 (Clothing, Electronics, Furniture, Groceries, Toys) | raw data |
| Store-product pairs simulated | 20 | `reports/simulation_results.csv` |
| Simulation horizon | 365 days | `src/simulation/inventory.py` |
| Inventory scenarios compared | 4 (Conservative, Current, Lean, EOQ Optimal) | NB04 |

---

## Project Structure

```
SupplychainoptimisationusingAI/
├── notebooks/
│   ├── 01_Data_Exploration_and_Cleaning.ipynb
│   ├── 02_Clustering_Feature_Engineering_ARM.ipynb
│   ├── 03_Forecasting_Models_and_Evaluation.ipynb
│   └── 04_Inventory_Simulation_Business_Metrics.ipynb
├── src/
│   ├── data/preprocessing.py
│   ├── features/engineering.py
│   ├── models/
│   │   ├── forecasting.py          # 7 models incl. LSTM
│   │   ├── evaluation.py           # MAE/RMSE/MAPE/sMAPE/Bias + rolling-origin backtest
│   │   └── quantile_forecast.py    # XGBoost quantile regression q10/q50/q90
│   ├── simulation/inventory.py     # EOQ, safety stock, ROP, newsvendor, simulation
│   └── analysis/horizons.py        # Multi-horizon target generation and evaluation
├── api/main.py                     # FastAPI: /health, /simulate, /policy/recommend
├── dashboard/app.py                # Streamlit: 4-tab dashboard, no sidebar
├── data/
│   ├── raw/retail_store_inventory.csv
│   └── processed/
│       ├── cleaned_retail_data.csv
│       └── feature_matrix.csv
├── models/
│   ├── xgboost_demand.pkl
│   ├── lightgbm_demand.pkl
│   ├── lstm_demand.pkl
│   └── feature_cols.pkl
├── reports/simulation_results.csv
├── .streamlit/config.toml
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 1. Data and Features

**Pipeline:** `notebooks/01` → `data/processed/cleaned_retail_data.csv` → `notebooks/02+03` → `data/processed/feature_matrix.csv`

**24 model features** (from `models/feature_cols.pkl`):

| Group | Features |
|---|---|
| Lag | `Units_Sold_lag_1`, `lag_7`, `lag_14`, `lag_28` |
| Rolling mean | `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_28` |
| Rolling std | `rolling_std_7`, `rolling_std_14`, `rolling_std_28` |
| Calendar | `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `week_of_year`, `is_weekend` |
| Cyclical | `month_sin`, `month_cos`, `dow_sin`, `dow_cos` |
| Market | `Demand_Forecast`, `Competitor_Pricing` |
| Inventory | `Inventory_Turnover` |

**Leakage prevention:** All rolling features use `.shift(1)` before `.rolling()` so day T's feature never includes day T's demand:
```python
df['rolling_mean_7'] = df.groupby(['store_id','product_id'])['Units_Sold'].shift(1).rolling(7).mean()
```

---

## 2. Forecasting Models

Seven models benchmarked on a strict **chronological 30-day holdout** (never random split):

| Model | Type | Notes |
|---|---|---|
| Naive Baseline | Statistical | predict t+1 = t |
| Moving Average (7d) | Statistical | baseline |
| ARIMA (1,1,1) | Time Series | per-series |
| Prophet | Time Series | additive seasonality |
| LightGBM | Tabular ML | leaf-wise boosting |
| **XGBoost** | **Tabular ML** | **best classical model** |
| LSTM | Deep Learning | PyTorch, seq_len=28 |

**XGBoost point forecast result:** ~15% MAPE on the 30-day holdout.
**LSTM result:** MAE=74.19, RMSE=86.95 on holdout.

**Rolling-origin backtesting** (`src/models/evaluation.py → rolling_origin_backtest()`):
3 expanding windows, 30-day test each:
```
Split 1:  MAE=9.68  RMSE=10.9  Bias=-0.87
Split 2:  MAE=7.60  RMSE=9.72  Bias=+0.19
Split 3:  MAE=5.63  RMSE=7.37  Bias=-0.36
Mean:     MAE=7.57  RMSE=9.33  Bias=-0.33
```

**Bias definition:** `predicted - actual`. Negative = model underforecasts on average.

---

## 3. Probabilistic Forecasting

**File:** `src/models/quantile_forecast.py`

XGBoost trained separately at three quantiles using `objective='reg:quantileerror'`:

| Quantile | Meaning | Pinball Loss |
|---|---|---|
| q10 | 10th percentile — demand floor | 3.74 |
| q50 | Median — central forecast | 6.08 |
| q90 | 90th percentile — demand ceiling | 3.09 |

**Note:** q10–q90 is an **80% prediction interval**, not 90%.

**Calibration check** (50 test samples):
```
q10: target=10% coverage, actual=18%
q50: target=50% coverage, actual=62%
q90: target=90% coverage, actual=84%
```

**Interval width:** Mean q90-q10 gap = 26.5 units (on synthetic test data during unit testing).

**Safety stock from quantile** (`safety_stock_from_quantile()`):
```python
daily_buffer = mean(q90 - q50)   # average upside uncertainty per day
safety_stock = daily_buffer * sqrt(lead_time)
# = 35 units at lead_time=7 days (on unit-test data)
```
This replaces the LSTM residual std proxy used in the initial version.

---

## 4. Evaluation Breakdown

**File:** `src/models/evaluation.py`

| Function | What it produces |
|---|---|
| `evaluate_all()` | MAE, RMSE, MAPE, sMAPE, Bias per model |
| `evaluate_by_group()` | Per store-product pair metrics |
| `evaluate_by_volume_segment()` | Metrics split by Low/Medium/High demand volume |
| `calibration_check()` | ±1σ/2σ/3σ actual vs expected coverage |
| `evaluate_multi_horizon()` | Metrics at 1-day, 7-day, 30-day horizons |
| `rolling_origin_backtest()` | Expanding-window CV, returns per-split + mean/std |

---

## 5. Inventory Formulas

**File:** `src/simulation/inventory.py`

**EOQ** (annualized demand):
```python
EOQ = sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit_per_year)
# annual_demand = avg_daily_demand * 365
```

**Safety Stock:**
```python
SS = Z * sigma * sqrt(lead_time)
# Z=1.65 for 95% service level
# sigma = demand std or quantile-derived uncertainty
```

**Reorder Point:**
```python
ROP = avg_daily_demand * lead_time + safety_stock
```

**Newsvendor critical fractile** (`newsvendor_order_quantity()`):
```python
q* = Cu / (Cu + Co)   # = 5/(5+1) = 0.8333
# Cu = underage cost (lost sale) = $5
# Co = overage cost (waste/holding) = $1
# Optimal order quantity = demand_mean + Z(0.833) * demand_std
# = 50 + 0.967 * 15 = 64.5 units
# q* is not trained — it is derived analytically and mapped to normal quantile
```

---

## 6. Inventory Simulation

**File:** `src/simulation/inventory.py → simulate_inventory()`

Day-by-day loop over actual historical demand (not forecasts):
- Each day: receive pending orders → subtract demand → check reorder point → place order if needed
- Stockout = day when inventory hits 0
- Service level = days with sufficient stock / 365

**Results from `reports/simulation_results.csv` (20 store-product pairs):**

| Metric | Value |
|---|---|
| Service Level | 99.77% (364.2 of 365 days fully met on average) |
| Stockout Rate | 0.23% (~0.84 days/year per pair) |
| Total cost range | $582k – $633k across pairs |

**Cost sensitivity analysis** (lead_time_days varied, same policy):
```
Lead time 3 days:   SL=100.0%   Total cost=$128k
Lead time 7 days:   SL=98.6%    Total cost=$220k
Lead time 14 days:  SL=93.4%    Total cost=$423k
```
Note: these cost figures use a single store-product pair with fixed policy parameters for illustration. The $623k figure in simulation results reflects the actual 365-day simulation over the full dataset with holding_cost_per_unit=0.5/day.

**4 inventory strategies:**
1. Conservative — Z=2.33, 99% service level target, large buffer
2. Current — Z=1.65, 95% service level target, baseline
3. Lean — Z=1.04, 85% service level target, minimal buffer
4. EOQ Optimal — math-derived order quantity, Z=1.65

---

## 7. Application

**Dashboard** (`dashboard/app.py`):
- 4 tabs: Demand Analysis / Forecast Models / Inventory Simulation / Scenario Planning
- No sidebar — inline filter bar (store, product, lead time, cost inputs)
- Light theme via `.streamlit/config.toml`
- Run locally: `streamlit run dashboard/app.py`
- **Not deployed publicly** — runs locally or via Docker

**FastAPI** (`api/main.py`):
- `GET /health` — liveness check
- `POST /simulate` — runs inventory simulation with custom policy parameters
- `POST /policy/recommend` — returns EOQ, safety stock, ROP for given demand inputs
- Docs at `http://localhost:8000/docs` when running

**Docker** (`docker-compose.yml`):
- Three services: `api` (port 8000), `dashboard` (port 8501), `db` (PostgreSQL 15)
- Run: `docker-compose up --build`

---

## 8. Known Limitations

| Gap | Honest statement |
|---|---|
| Probabilistic calibration | q90 achieved 84% coverage vs 90% target on unit-test data — model is slightly miscalibrated on small samples |
| Hierarchical forecasting | Forecasts are per store-product pair only — no reconciliation across store/region/chain levels |
| Rolling-origin metrics | The MAE=7.57 mean comes from synthetic random data in unit tests, not the actual retail dataset |
| Quantile safety stock | The 35-unit figure is from unit-test synthetic data — actual values depend on the fitted model's interval width on real data |
| Deployment | Application runs locally and via Docker — no public cloud deployment |
| Censored demand | Stockout days record 0 sales even when demand existed — model training does not correct for this |

---

## Setup

```bash
# Local
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
streamlit run dashboard/app.py

# Docker
docker-compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## Dataset

[Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) — Kaggle

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML / Forecasting | pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, prophet, pytorch |
| Probabilistic | XGBoost quantile regression (`objective='reg:quantileerror'`) |
| API | fastapi, uvicorn, pydantic |
| Dashboard | streamlit, plotly |
| Database | postgresql, sqlalchemy |
| Infrastructure | docker, docker-compose |
