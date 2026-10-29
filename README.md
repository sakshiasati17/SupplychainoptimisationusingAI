# Supply Chain Demand Forecasting & Inventory Optimization

End-to-end retail inventory decision system: raw sales data → probabilistic forecasts → inventory simulation → REST API → dashboard.

---

## Verified Numbers

| Fact | Value | Verified From |
|---|---|---|
| Raw records | 73,185 | `data/raw/retail_store_inventory.csv` |
| Cleaned records | 72,287 | `data/processed/cleaned_retail_data.csv` |
| Feature matrix rows | 70,209 | `data/processed/feature_matrix.csv` |
| Model features (leakage-free Option B) | 25 | `scripts/fresh_numbers.py` |
| Stores | 5 | raw data |
| Products | 21 | raw data |
| Categories | 5 | raw data |
| Store-product pairs simulated | 100 | `reports/fresh_numbers/5_simulation_by_pair.csv` |

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
├── reports/
│   ├── fresh_numbers/               # verified evaluation and simulation outputs
│   └── simulation_results.csv
├── scripts/
│   └── fresh_numbers.py             # script to reproduce all evaluation metrics
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Feature Engineering

To prevent lookahead bias and target leakage, predictive features are strictly partitioned:

| Feature Set | Composition |
|---|---|
| **Lags & Rolling** | `Units_Sold_lag_1`, `lag_7`, `lag_14`, `lag_28`, rolling means & std (7d, 14d, 28d) |
| **Calendar & Cyclical** | `day_of_week`, `day_of_month`, `month`, `quarter`, `year`, `week_of_year`, `is_weekend`, sine/cosine transforms |
| **Exogenous Plans** | `Price`, `Discount`, `Holiday_Promotion`, `Competitor_Pricing` |
| **Option B (Headline)** | Lags + Rolling + Calendar + Exogenous (25 leakage-free features) |
| **Option A (Comparison)** | Option B + dataset's `Demand_Forecast` column |

*Note: Same-day variables (`Inventory_Turnover`, `Inventory_Level`, `Units_Ordered`, gap, and stockout flags) are excluded from predictive feature sets to avoid target leakage.*

---

## Forecasting & Model Evaluation

### 1. Global Models (30-day holdout: 67,211 train / 2,998 test rows)

| Model | MAE | RMSE | sMAPE | Bias |
|---|---|---|---|---|
| **XGBoost B (Leakage-free)** | **87.68** | **108.57** | **71.42%** | **-3.20** |
| Moving Average (28d) | 88.86 | 110.43 | 71.60% | 0.76 |
| Moving Average (7d) | 92.39 | 115.91 | 73.56% | 0.20 |
| Naive (yesterday) | 119.89 | 154.94 | 93.07% | -0.04 |
| *Dataset Demand_Forecast as-is* | *9.25* | *29.29* | *17.22%* | *3.73* |
| *XGBoost A (+ Demand_Forecast)* | *8.31* | *21.62* | *14.33%* | *-0.91* |

- XGBoost B improves error by **27% over Naive** (87.68 vs 119.89 MAE) without data leakage.
- In Option A, `Demand_Forecast` accounts for 87% of feature importance, driving the low error (~8.3 MAE); Option B serves as the honest, leakage-free benchmark.

### 2. Single-Series Benchmark (Store S001 / Product P0001, 30-day forecast)

| Model | MAE | RMSE | sMAPE | Bias |
|---|---|---|---|---|
| **ARIMA(1,1,1)** | **71.05** | **84.38** | **60.77%** | **9.72** |
| **LSTM (PyTorch, 200 epochs)** | **71.25** | **84.49** | **60.86%** | **10.66** |
| Moving Average (7d) | 75.08 | 87.94 | 62.28% | 26.60 |
| LSTM (PyTorch, 20 epochs) | 77.20 | 90.60 | 63.01% | 34.24 |
| Naive | 96.97 | 111.92 | 70.35% | 74.17 |

- With sufficient training (200 epochs), LSTM effectively ties ARIMA (71.25 vs 71.05 MAE).

### 3. Probabilistic Forecasting (XGBoost Quantile Regression)

Quantile regression models trained with pinball loss at q10, q50, and q90:

- **Option B (Leakage-free):** 80% prediction interval (q10 to q90) achieves **78.7% empirical coverage** (target: 80.0%), with pinball losses q10: 12.76, q50: 43.00, q90: 22.54, and mean interval width of 267.0 units.
- **Option A:** 76.8% coverage, mean width 25.7 units.

### 4. Rolling-Origin Backtest (3 expanding windows of ~30 days)

- **Option B (Leakage-free):** Mean MAE = **89.02 ± 1.70**, RMSE = 109.43 ± 0.97, sMAPE = 71.62% ± 0.93%, Bias = -0.98 ± 1.70. Demonstrates consistent error across multiple time horizons without temporal leakage.
- **Option A:** Mean MAE = 7.88 ± 0.33, Bias = -0.14 ± 0.70.

---

## Inventory Simulation & Optimization

Inventory formulas implemented in `src/simulation/inventory.py`:
- **EOQ** = $\sqrt{\frac{2 \times \text{annual\_demand} \times S}{H}}$
- **Safety Stock** = $Z \times \sigma \times \sqrt{L}$
- **Reorder Point (ROP)** = $\bar{d} \times L + \text{Safety Stock}$

### Reorder Logic Fix & Impact

The original simulator evaluated reorder decisions against on-hand stock only, ignoring orders placed but still in transit. During lead time ($L=3$ days), this caused continuous reordering and inflated inventory to 1,680 units (95.4% overstock rate).

Accounting for **Inventory Position** ($\text{On-Hand} + \text{On-Order Pipeline}$) prevents duplicate reordering and **cuts average inventory by 51.6%** while maintaining a **99.22% service level**.

### 100-Pair Simulation Results (Last 180 Days, $L=3$, $Z=1.65$)

| Policy | Service Level | Stockout Rate | Overstock Rate | Avg Inventory | Total Cost | Inv vs Policy 1 |
|---|---|---|---|---|---|---|
| **0. Legacy logic (reorder on on-hand)** | 99.73% | 0.27% | 95.36% | 1,680.13 | $15,144,103 | +106.7% |
| **1. Fixed simulator (static ROP + demand $\sigma$ SS)** | **99.22%** | **0.78%** | **87.01%** | **812.89** | **$7,381,924** | **0.0% (Baseline)** |
| **2. Moving-avg forecast + error $\sigma$ SS** | 99.25% | 0.75% | 87.39% | 828.16 | $7,514,953 | +1.9% |
| **3. Demand_Forecast + error $\sigma$ SS** | 99.93% | 0.07% | 77.49% | 556.98 | $5,022,651 | -31.5% |

- Fixing the pipeline order tracking cuts operating costs by over 50% ($15.1M down to $7.38M).
- Overstock rates remain ~87% across fixed policies because EOQ batch sizing (~7 days of supply) exceeds the $2 \times \text{daily demand} + 1$ threshold.

---

## Application & Usage

### FastAPI Endpoints (`api/main.py`)
- `GET  /health` - API health check
- `POST /simulate` - Run inventory simulation with customizable policy
- `POST /policy/recommend` - Calculate EOQ, Safety Stock, and ROP

### Streamlit Dashboard (`dashboard/app.py`)
4 interactive tabs:
1. **Demand Analysis** - Historical sales patterns and moving averages
2. **Forecast Models** - Multi-model comparison and predictions
3. **Inventory Simulation** - Interactive ROP/EOQ simulation with pipeline tracking
4. **Scenario Planning** - Stress testing (Conservative, Current, Lean, EOQ Optimal)

### Reproduce Fresh Numbers
```bash
python scripts/fresh_numbers.py
```
Outputs are saved to `reports/fresh_numbers/`.

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn api.main:app --reload --port 8000

# Start Dashboard
streamlit run dashboard/app.py
```

### Docker
```bash
docker-compose up --build
```

---

## Tech Stack

pandas · numpy · scikit-learn · xgboost · lightgbm · statsmodels · prophet · pytorch · fastapi · streamlit · plotly · docker · postgresql
