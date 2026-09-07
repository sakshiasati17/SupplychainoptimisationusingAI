# Supply Chain Demand Forecasting & Inventory Optimization

End-to-end retail inventory decision system — raw sales data → probabilistic forecasts → inventory simulation → REST API → dashboard.

---

## Verified Numbers

| Fact | Value | Verified From |
|---|---|---|
| Raw records | 73,185 | `data/raw/retail_store_inventory.csv` |
| Cleaned records | 72,287 | `data/processed/cleaned_retail_data.csv` |
| Feature matrix rows | 70,209 | `data/processed/feature_matrix.csv` |
| Model features | 24 | `models/feature_cols.pkl` |
| Stores | 5 | raw data |
| Products | 21 | raw data |
| Categories | 5 | raw data |
| Store-product pairs simulated | 20 | `reports/simulation_results.csv` |

---

## Project Structure

```
├── notebooks/
│   ├── 01_Data_Exploration_and_Cleaning.ipynb
│   ├── 02_Clustering_Feature_Engineering_ARM.ipynb
│   ├── 03_Forecasting_Models_and_Evaluation.ipynb
│   └── 04_Inventory_Simulation_Business_Metrics.ipynb
├── src/
│   ├── data/preprocessing.py
│   ├── features/engineering.py
│   ├── models/
│   │   ├── forecasting.py           # 7 models incl. LSTM
│   │   ├── evaluation.py            # metrics + rolling-origin backtest
│   │   └── quantile_forecast.py     # XGBoost q10/q50/q90
│   ├── simulation/inventory.py      # EOQ, safety stock, ROP, newsvendor, simulation
│   └── analysis/horizons.py         # multi-horizon targets and evaluation
├── api/main.py                      # FastAPI endpoints
├── dashboard/app.py                 # Streamlit 4-tab dashboard
├── models/                          # saved .pkl artifacts
├── data/processed/                  # cleaned CSV + feature matrix
├── reports/simulation_results.csv
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Features (24 model features)

| Group | Features |
|---|---|
| Lag | lag_1, lag_7, lag_14, lag_28 |
| Rolling mean | 7d, 14d, 28d (shift-before-rolling — no leakage) |
| Rolling std | 7d, 14d, 28d |
| Calendar | day_of_week, day_of_month, month, quarter, year, week_of_year, is_weekend |
| Cyclical | month_sin, month_cos, dow_sin, dow_cos |
| Market | Demand_Forecast, Competitor_Pricing |
| Inventory | Inventory_Turnover |

---

## Models

7 models on a strict chronological 30-day holdout (no random split):

| Model | Notes |
|---|---|
| Naive, Moving Average | baselines |
| ARIMA, Prophet | statistical time series |
| LightGBM, **XGBoost** | **best classical — ~15% MAPE** |
| LSTM (PyTorch) | MAE=74.19, RMSE=86.95 |

**Probabilistic:** XGBoost quantile regression at q10/q50/q90 (`objective='reg:quantileerror'`, pinball loss). q10–q90 = 80% interval.

**Rolling-origin backtest:** 3 expanding windows — mean MAE=7.57, Bias=-0.33 (predicted − actual).

---

## Inventory

All formulas in `src/simulation/inventory.py`:

- **EOQ** = √(2 × annual\_demand × S / H)
- **Safety Stock** = Z × σ × √lead\_time
- **ROP** = avg\_daily\_demand × lead\_time + safety\_stock
- **Newsvendor** q\* = Cu / (Cu + Co) = 5/6 = 0.833 → order qty = 64.5 units

**365-day simulation** across 20 store-product pairs:
- Service level: **99.77%** (day-level availability)
- Stockout rate: **0.23%**
- 4 strategies compared: Conservative / Current / Lean / EOQ Optimal

---

## Application

**FastAPI** (`api/main.py`): `/health`, `/simulate`, `/policy/recommend`

**Streamlit** (`dashboard/app.py`): 4 tabs — Demand Analysis / Forecast Models / Inventory Simulation / Scenario Planning. Runs locally, not publicly deployed.

**Docker** (`docker-compose.yml`): API (8000) + Dashboard (8501) + PostgreSQL (5432)

```bash
# Local
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
streamlit run dashboard/app.py

# Docker
docker-compose up --build
```

---

## Dataset

[Retail Store Inventory Forecasting — Kaggle](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)

## Tech Stack

pandas · numpy · scikit-learn · xgboost · lightgbm · statsmodels · prophet · pytorch · fastapi · streamlit · plotly · docker · postgresql
