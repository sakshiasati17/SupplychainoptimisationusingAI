"""
Streamlit dashboard for Supply Chain Demand Forecasting & Inventory Simulation.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.simulation.inventory import (
    InventoryPolicy,
    simulate_inventory,
    compute_eoq,
    compute_safety_stock,
    compute_reorder_point,
    run_scenario_comparison,
)
from src.models.evaluation import evaluate_all, compare_models

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain AI Dashboard",
    page_icon="🏭",
    layout="wide",
)

st.title("🏭 Supply Chain Demand Forecasting & Inventory Simulation")
st.markdown("Built with ML forecasting models + business cost simulation.")

# ---------------------------------------------------------------------------
# Sidebar — Data & Policy Controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

DATA_PATH = Path("data/processed/cleaned_retail_data.csv")

@st.cache_data
def load_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        return df
    # Fallback: synthetic data for demo
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=365)
    df = pd.DataFrame({
        "Date": dates,
        "Store_ID": "S001",
        "Product_ID": "P0001",
        "Units_Sold": np.random.poisson(50, 365) + 10 * np.sin(np.arange(365) * 2 * np.pi / 365),
        "Inventory_Level": np.random.randint(80, 200, 365),
    })
    return df

df = load_data()

stores = sorted(df["Store_ID"].unique()) if "Store_ID" in df.columns else ["S001"]
products = sorted(df["Product_ID"].unique()) if "Product_ID" in df.columns else ["P0001"]

selected_store = st.sidebar.selectbox("Store", stores)
selected_product = st.sidebar.selectbox("Product", products)

# Filter data
mask = (df["Store_ID"] == selected_store) & (df["Product_ID"] == selected_product)
df_filtered = df[mask].sort_values("Date") if mask.any() else df.sort_values("Date")

st.sidebar.markdown("---")
st.sidebar.subheader("Inventory Policy")
reorder_point = st.sidebar.slider("Reorder Point (units)", 10, 200, 50)
order_quantity = st.sidebar.slider("Order Quantity (units)", 50, 500, 150)
lead_time = st.sidebar.slider("Lead Time (days)", 1, 14, 3)
holding_cost = st.sidebar.number_input("Holding Cost ($/unit/day)", 0.1, 5.0, 0.5, step=0.1)
stockout_cost = st.sidebar.number_input("Stockout Cost ($/unit lost)", 1.0, 20.0, 5.0, step=0.5)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Data Overview", "📈 Demand Forecast", "🏗️ Inventory Simulation", "💰 Cost Comparison"]
)

# ---- Tab 1: Data Overview ----
with tab1:
    st.subheader(f"Store: {selected_store} | Product: {selected_product}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df_filtered):,}")
    if "Units_Sold" in df_filtered.columns:
        col2.metric("Avg Daily Demand", f"{df_filtered['Units_Sold'].mean():.1f}")
        col3.metric("Max Daily Demand", f"{df_filtered['Units_Sold'].max():.0f}")
        col4.metric("Demand Std Dev", f"{df_filtered['Units_Sold'].std():.1f}")

    if "Units_Sold" in df_filtered.columns and "Date" in df_filtered.columns:
        fig = px.line(
            df_filtered,
            x="Date",
            y="Units_Sold",
            title="Historical Demand (Units Sold)",
            labels={"Units_Sold": "Units Sold"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Inventory_Level" in df_filtered.columns:
        fig2 = px.area(
            df_filtered,
            x="Date",
            y="Inventory_Level",
            title="Inventory Level Over Time",
            color_discrete_sequence=["#636EFA"],
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---- Tab 2: Demand Forecast ----
with tab2:
    st.subheader("Demand Forecasting with Moving Average & Naive Baseline")

    if "Units_Sold" in df_filtered.columns and len(df_filtered) > 30:
        series = df_filtered.set_index("Date")["Units_Sold"]

        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)
        ma_window = st.slider("Moving Average Window", 3, 30, 7)

        # Moving average forecast on last 'horizon' days
        ma_forecast = series.rolling(ma_window, min_periods=1).mean().values[-horizon:]
        naive_forecast = np.full(horizon, series.iloc[-1])
        future_dates = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=horizon)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index[-60:], y=series.values[-60:], name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=future_dates, y=ma_forecast, name=f"Moving Avg (w={ma_window})", line=dict(dash="dash", color="green")))
        fig.add_trace(go.Scatter(x=future_dates, y=naive_forecast, name="Naive Baseline", line=dict(dash="dot", color="red")))
        fig.update_layout(title="Demand Forecast", xaxis_title="Date", yaxis_title="Units Sold")
        st.plotly_chart(fig, use_container_width=True)

        # Model evaluation on train/test split
        split = int(len(series) * 0.8)
        actual = series.values[split:]
        ma_pred = series.rolling(ma_window, min_periods=1).mean().values[split:]
        naive_pred = series.shift(1).values[split:]
        naive_pred = np.where(np.isnan(naive_pred), series.mean(), naive_pred)

        results = [
            evaluate_all(actual, naive_pred, "Naive Baseline"),
            evaluate_all(actual, ma_pred, f"Moving Average (w={ma_window})"),
        ]
        st.subheader("Model Evaluation Metrics")
        st.dataframe(compare_models(results).style.highlight_min(axis=0, color="lightgreen"))
    else:
        st.info("Not enough data for the selected store/product combination.")

# ---- Tab 3: Inventory Simulation ----
with tab3:
    st.subheader("Inventory Simulation")

    if "Units_Sold" in df_filtered.columns and len(df_filtered) > 10:
        demand = df_filtered["Units_Sold"].values
        policy = InventoryPolicy(
            reorder_point=reorder_point,
            order_quantity=order_quantity,
            lead_time_days=lead_time,
            holding_cost_per_unit=holding_cost,
            stockout_cost_per_unit=stockout_cost,
            initial_inventory=df_filtered["Inventory_Level"].iloc[0] if "Inventory_Level" in df_filtered.columns else 100,
        )

        result = simulate_inventory(demand, policy)
        summary = result.summary()

        col1, col2, col3 = st.columns(3)
        col1.metric("Stockout Rate", f"{summary['stockout_rate_%']}%", delta_color="inverse")
        col2.metric("Service Level", f"{summary['service_level_%']}%")
        col3.metric("Total Operating Cost", f"${summary['total_operating_cost_$']:,.2f}", delta_color="inverse")

        col4, col5, col6 = st.columns(3)
        col4.metric("Holding Cost", f"${summary['total_holding_cost_$']:,.2f}")
        col5.metric("Lost Sales Cost", f"${summary['total_lost_sales_cost_$']:,.2f}")
        col6.metric("Orders Placed", f"{summary['n_orders_placed']}")

        # Inventory level chart
        sim_df = pd.DataFrame({
            "Day": range(len(result.daily_inventory)),
            "Inventory": result.daily_inventory,
            "Demand": result.daily_demand,
        })
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim_df["Day"], y=sim_df["Inventory"], name="Inventory Level", fill="tozeroy"))
        fig.add_trace(go.Scatter(x=sim_df["Day"], y=sim_df["Demand"], name="Demand", line=dict(dash="dash", color="red")))
        fig.add_hline(y=reorder_point, line_dash="dot", annotation_text="Reorder Point", line_color="orange")
        fig.update_layout(title="Simulated Inventory vs Demand", xaxis_title="Day", yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)

# ---- Tab 4: Cost Comparison ----
with tab4:
    st.subheader("What-If Scenario Cost Comparison")

    if "Units_Sold" in df_filtered.columns:
        demand = df_filtered["Units_Sold"].values
        avg_d = np.mean(demand)
        std_d = np.std(demand)

        scenarios = {
            "Conservative (High ROP)": InventoryPolicy(
                reorder_point=avg_d * 3, order_quantity=avg_d * 7,
                lead_time_days=lead_time, holding_cost_per_unit=holding_cost,
                stockout_cost_per_unit=stockout_cost, initial_inventory=avg_d * 4,
            ),
            "Current Policy": InventoryPolicy(
                reorder_point=reorder_point, order_quantity=order_quantity,
                lead_time_days=lead_time, holding_cost_per_unit=holding_cost,
                stockout_cost_per_unit=stockout_cost,
                initial_inventory=df_filtered["Inventory_Level"].iloc[0] if "Inventory_Level" in df_filtered.columns else 100,
            ),
            "Lean (Low ROP)": InventoryPolicy(
                reorder_point=avg_d * 1, order_quantity=avg_d * 3,
                lead_time_days=lead_time, holding_cost_per_unit=holding_cost,
                stockout_cost_per_unit=stockout_cost, initial_inventory=avg_d * 2,
            ),
            "EOQ-Optimized": InventoryPolicy(
                reorder_point=compute_reorder_point(avg_d, lead_time, compute_safety_stock(avg_d, std_d, lead_time)),
                order_quantity=compute_eoq(avg_d * 365),
                lead_time_days=lead_time, holding_cost_per_unit=holding_cost,
                stockout_cost_per_unit=stockout_cost, initial_inventory=avg_d * 3,
            ),
        }

        comparison_df = run_scenario_comparison(demand, scenarios)
        st.dataframe(comparison_df.style.highlight_min(subset=["total_operating_cost_$"], color="lightgreen"))

        fig = px.bar(
            comparison_df.reset_index(),
            x="scenario",
            y=["total_holding_cost_$", "total_lost_sales_cost_$"],
            barmode="stack",
            title="Total Cost Breakdown by Scenario",
            labels={"value": "Cost ($)", "variable": "Cost Type"},
            color_discrete_map={"total_holding_cost_$": "#636EFA", "total_lost_sales_cost_$": "#EF553B"},
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            comparison_df.reset_index(),
            x="stockout_rate_%",
            y="total_operating_cost_$",
            text="scenario",
            size="total_units_ordered",
            title="Stockout Rate vs Total Operating Cost",
            labels={"stockout_rate_%": "Stockout Rate (%)", "total_operating_cost_$": "Total Cost ($)"},
        )
        fig2.update_traces(textposition="top center")
        st.plotly_chart(fig2, use_container_width=True)
