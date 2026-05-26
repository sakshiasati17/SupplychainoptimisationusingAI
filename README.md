# Supply Chain Optimization & Demand Analysis

**Business-Driven Inventory Decision Support System**

> "Developed a data-driven system to optimize retail inventory levels by analyzing historical demand patterns and simulating cost-effective reorder strategies."

## Project Overview

This project focuses on **operational decision-making** within a retail supply chain. By analyzing inventory data across 73 stores, I built a system that identifies stockout risks, quantifies overstock costs, and recommends optimal reorder quantities. The goal is to balance **Holding Costs** against **Service Levels** to drive better business outcomes.


---

## Project Structure

```
SupplychainoptimisationusingAI/
├── notebooks/
│   ├── 01_Data_Exploration_and_Cleaning.ipynb         # EDA, data cleaning, visualizations
│   ├── 02_Clustering_Feature_Engineering_ARM.ipynb    # PCA, K-Means, DBSCAN, Association Rules
│   ├── 03_Forecasting_Models_and_Evaluation.ipynb     # ARIMA, Prophet, XGBoost, LightGBM + MAE/RMSE/MAPE
│   └── 04_Inventory_Simulation_Business_Metrics.ipynb # SVM/RF + Inventory Simulation + Cost Analysis
├── src/
│   ├── data/
│   │   └── preprocessing.py      # Data cleaning pipeline
│   ├── features/
│   │   └── engineering.py        # Lag, rolling, time features
│   ├── models/
│   │   ├── forecasting.py        # ARIMA, Prophet, XGBoost, LightGBM, LSTM
│   │   └── evaluation.py         # MAE, RMSE, MAPE, sMAPE
│   └── simulation/
│       └── inventory.py          # EOQ, safety stock, reorder simulation
├── api/
│   └── main.py                   # FastAPI REST endpoints
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── data/
│   ├── raw/                      # Original retail inventory CSV
│   └── processed/                # Cleaned data + association rules
├── models/                       # Saved model artifacts
├── scripts/                      # Training and simulation scripts
├── reports/                      # Analysis reports
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Notebooks

| # | Notebook | Content |
|---|---|---|
| 01 | Data Exploration & Cleaning | EDA, null handling, outlier removal, temporal analysis |
| 02 | Clustering, Feature Engineering & ARM | PCA, K-Means, Hierarchical, DBSCAN, Association Rule Mining |
| 03 | Forecasting Models & Evaluation | Naive, MA, ARIMA, Prophet, XGBoost, LightGBM — MAE/RMSE/MAPE/sMAPE |
| 04 | Inventory Simulation & Business Metrics | SVM/RF, EOQ, safety stock, stockout/overstock simulation, scenario comparison |

---

## Models Compared

| Model | Type |
|---|---|
| Naive Baseline | Statistical |
| Moving Average | Statistical |
| ARIMA / SARIMA | Time Series |
| Prophet | Time Series |
| XGBoost | Tabular ML |
| LightGBM | Tabular ML |
| LSTM | Deep Learning |

---

## Business Metrics

| Metric | Description |
|---|---|
| Stockout Rate | % of days demand exceeded inventory |
| Overstock Rate | % of days inventory > 2x demand |
| Service Level | % of days demand fully met |
| Holding Cost | Cost of keeping unsold inventory |
| Lost Sales Cost | Revenue lost due to stockouts |
| Total Operating Cost | Holding + Lost Sales |
| EOQ | Economic Order Quantity |
| Safety Stock | Buffer against demand variability |

---

## Forecast Evaluation Metrics

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Square Error
- **MAPE** — Mean Absolute Percentage Error
- **sMAPE** — Symmetric MAPE

---

## Setup & Run

### Local

```bash
pip install -r requirements.txt

# Run API
uvicorn api.main:app --reload --port 8000

# Run Dashboard
streamlit run dashboard/app.py
```

### Docker

```bash
docker-compose up --build
```

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## Dataset

- **Source**: [Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) (Kaggle)
- **Size**: ~73,000 rows, 73 stores, 20 products, 4 regions
- **Features**: Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Price, Discount, Weather Condition, Seasonality, Competitor Pricing

---

## Tech Stack

| Layer | Libraries |
|---|---|
| ML / Forecasting | pandas, numpy, scikit-learn, xgboost, lightgbm, statsmodels, prophet, pytorch |
| API | fastapi, uvicorn, pydantic |
| Dashboard | streamlit, plotly |
| Database | postgresql, sqlalchemy, psycopg2 |
| Infrastructure | docker, docker-compose |
