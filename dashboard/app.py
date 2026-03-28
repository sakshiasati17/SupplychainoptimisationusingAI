"""
Supply Chain AI — Demand Forecasting & Inventory Simulation Dashboard
Run: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.simulation.inventory import (
    InventoryPolicy, simulate_inventory,
    compute_eoq, compute_safety_stock, compute_reorder_point,
    run_scenario_comparison,
)
from src.models.evaluation import evaluate_all, compare_models

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ───────────────────────────────────────────────────────────
BLUE   = "#2563EB"
INDIGO = "#4F46E5"
GREEN  = "#059669"
RED    = "#DC2626"
AMBER  = "#D97706"
GREY   = "#6B7280"
BG     = "#F9FAFB"

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #F9FAFB; }
[data-testid="stSidebar"] { background: #1E293B; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94A3B8 !important; font-size:0.78rem; }
h1 { font-size: 1.5rem !important; font-weight: 700 !important; color: #0F172A; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1E293B; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #374151; }
.metric-card {
    background: white; border-radius: 10px; padding: 1rem 1.2rem;
    border: 1px solid #E5E7EB; box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.metric-label { font-size: 0.72rem; color: #6B7280; font-weight: 600;
                text-transform: uppercase; letter-spacing: .05em; }
.metric-value { font-size: 1.6rem; font-weight: 700; color: #0F172A; margin-top:.2rem; }
.metric-sub   { font-size: 0.78rem; color: #6B7280; margin-top:.1rem; }
div[data-testid="stTab"] button { font-size: 0.85rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    p = Path("data/processed/cleaned_retail_data.csv")
    if not p.exists():
        p = Path("data/raw/retail_store_inventory.csv")
    df = pd.read_csv(p, parse_dates=["Date"])
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    return df

@st.cache_resource
def load_models():
    out = {}
    for name, path in [("xgboost",  "models/xgboost_demand.pkl"),
                        ("lightgbm", "models/lightgbm_demand.pkl"),
                        ("lstm",     "models/lstm_demand.pkl"),
                        ("features", "models/feature_cols.pkl")]:
        p = Path(path)
        if p.exists():
            out[name] = joblib.load(p)
    return out

df = load_data()
models = load_models()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Supply Chain AI")
    st.markdown("---")

    stores   = sorted(df["Store_ID"].unique())
    products = sorted(df["Product_ID"].unique())
    store   = st.selectbox("Store", stores)
    product = st.selectbox("Product", products)

    st.markdown("---")
    st.markdown("**Inventory Policy**")
    rop_slider = st.slider("Reorder Point (units)", 10, 500, 100)
    eoq_slider = st.slider("Order Quantity (units)", 50, 1000, 300)
    lead_time  = st.slider("Lead Time (days)", 1, 14, 3)
    h_cost     = st.number_input("Holding Cost ($/unit/day)", 0.1, 5.0, 0.5, 0.1)
    s_cost     = st.number_input("Stockout Cost ($/unit lost)", 1.0, 20.0, 5.0, 0.5)

    st.markdown("---")
    st.caption("Sakshi Asati · Supply Chain AI")

# ── Filter ───────────────────────────────────────────────────────────────────
mask = (df["Store_ID"] == store) & (df["Product_ID"] == product)
df_f = df[mask].sort_values("Date") if mask.any() else df.sort_values("Date")
ts   = df_f.groupby("Date")["Units_Sold"].sum().sort_index()
demand = ts.values

# ── KPIs ─────────────────────────────────────────────────────────────────────
avg_d = demand.mean(); std_d = demand.std()
eoq   = compute_eoq(avg_d * 365)
ss    = compute_safety_stock(avg_d, std_d, lead_time)
rop   = compute_reorder_point(avg_d, lead_time, ss)

st.markdown(f"## Supply Chain Dashboard &nbsp;·&nbsp; {store} / {product}")
st.markdown("---")

c1, c2, c3, c4, c5 = st.columns(5)
def metric_card(col, label, value, sub=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

metric_card(c1, "Avg Daily Demand", f"{avg_d:.0f} units", f"Std ±{std_d:.0f}")
metric_card(c2, "EOQ",              f"{eoq:.0f} units",   "Economic order qty")
metric_card(c3, "Safety Stock",     f"{ss:.0f} units",    f"Lead time {lead_time}d")
metric_card(c4, "Reorder Point",    f"{rop:.0f} units",   "95% service level")
metric_card(c5, "Data Points",      f"{len(ts):,}",       f"{ts.index[0].date()} – {ts.index[-1].date()}")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Demand", "🔮 Forecast", "🏭 Simulation", "💰 Scenarios"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Historical Demand
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.values, name="Daily Demand",
            line=dict(color=BLUE, width=1.5), fill="tozeroy",
            fillcolor="rgba(37,99,235,0.06)"
        ))
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.rolling(7).mean(),
            name="7-day MA", line=dict(color=AMBER, width=2, dash="dot")
        ))
        fig.update_layout(
            title="Historical Demand", height=320,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(orientation="h", y=1.1),
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F3F4F6"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Demand by category
        if "Category" in df_f.columns:
            cat = df_f.groupby("Category")["Units_Sold"].mean().sort_values(ascending=True)
            fig2 = go.Figure(go.Bar(
                x=cat.values, y=cat.index, orientation="h",
                marker_color=INDIGO, marker_line_width=0,
            ))
            fig2.update_layout(
                title="Avg Demand by Category", height=320,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=0, r=0, t=36, b=0),
                xaxis=dict(gridcolor="#F3F4F6"), yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Monthly heatmap
    if len(ts) > 60:
        heat_df = ts.reset_index()
        heat_df.columns = ["Date", "Units_Sold"]
        heat_df["Month"] = heat_df["Date"].dt.strftime("%b")
        heat_df["Year"]  = heat_df["Date"].dt.year
        pivot = heat_df.pivot_table(index="Year", columns="Month", values="Units_Sold", aggfunc="mean")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[m for m in month_order if m in pivot.columns]]

        fig3 = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index,
            colorscale=[[0,"#EFF6FF"],[1, BLUE]],
            showscale=True, hoverongaps=False,
        ))
        fig3.update_layout(
            title="Monthly Demand Heatmap", height=200,
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=36, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Forecast
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, key="horizon")
    TEST    = min(30, len(ts) // 4)
    train_s, test_s = ts.iloc[:-TEST], ts.iloc[-TEST:]
    actual  = test_s.values
    future_dates = pd.date_range(ts.index[-1] + pd.Timedelta(days=1), periods=horizon)
    results = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index[-90:], y=ts.values[-90:], name="Actual",
        line=dict(color="#1E293B", width=1.5)
    ))

    # Naive
    naive_pred = np.full(TEST, train_s.iloc[-1])
    results.append(evaluate_all(actual, naive_pred, "Naive Baseline"))
    fig.add_trace(go.Scatter(
        x=test_s.index, y=naive_pred, name="Naive",
        line=dict(color=GREY, dash="dot", width=1.5)
    ))

    # Moving Average
    ma_pred = np.full(TEST, train_s.rolling(7).mean().iloc[-1])
    results.append(evaluate_all(actual, ma_pred, "Moving Average"))
    fig.add_trace(go.Scatter(
        x=test_s.index, y=ma_pred, name="Moving Avg",
        line=dict(color=AMBER, dash="dot", width=1.5)
    ))

    # XGBoost
    if "xgboost" in models and "features" in models:
        try:
            feat_path = Path("data/processed/feature_matrix.csv")
            if feat_path.exists():
                feat_df = pd.read_csv(feat_path, parse_dates=["Date"])
                feat_df = feat_df[(feat_df["Store_ID"]==store) & (feat_df["Product_ID"]==product)].sort_values("Date")
                FCOLS = models["features"]
                FCOLS_avail = [c for c in FCOLS if c in feat_df.columns]
                if len(feat_df) > TEST and FCOLS_avail:
                    xgb_pred = models["xgboost"].predict(feat_df.tail(TEST)[FCOLS_avail])
                    results.append(evaluate_all(actual[:len(xgb_pred)], xgb_pred, "XGBoost"))
                    fig.add_trace(go.Scatter(
                        x=test_s.index[:len(xgb_pred)], y=xgb_pred, name="XGBoost",
                        line=dict(color=GREEN, width=2)
                    ))
                    lgb_pred = models["lightgbm"].predict(feat_df.tail(TEST)[FCOLS_avail])
                    results.append(evaluate_all(actual[:len(lgb_pred)], lgb_pred, "LightGBM"))
                    fig.add_trace(go.Scatter(
                        x=test_s.index[:len(lgb_pred)], y=lgb_pred, name="LightGBM",
                        line=dict(color=INDIGO, width=2)
                    ))
        except Exception:
            pass

    # LSTM with uncertainty band
    if "lstm" in models:
        try:
            lstm_pred = models["lstm"].predict(train_s, steps=TEST)
            results.append(evaluate_all(actual[:len(lstm_pred)], lstm_pred, "LSTM"))
            std_band = np.std(actual[:len(lstm_pred)] - lstm_pred)
            fig.add_trace(go.Scatter(
                x=test_s.index[:len(lstm_pred)], y=lstm_pred + std_band,
                line=dict(width=0), showlegend=False, name="LSTM upper"
            ))
            fig.add_trace(go.Scatter(
                x=test_s.index[:len(lstm_pred)], y=lstm_pred - std_band,
                fill="tonexty", fillcolor="rgba(220,38,38,0.08)",
                line=dict(width=0), name="LSTM ±1σ"
            ))
            fig.add_trace(go.Scatter(
                x=test_s.index[:len(lstm_pred)], y=lstm_pred, name="LSTM",
                line=dict(color=RED, width=2)
            ))
        except Exception:
            pass

    fig.add_vline(x=str(test_s.index[0]), line_dash="dash", line_color=GREY, line_width=1)
    fig.update_layout(
        title=f"Demand Forecast — {TEST}-day test window",
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)

    if results:
        st.markdown("#### Model Comparison")
        cmp = compare_models(results)
        st.dataframe(
            cmp.style
               .format("{:.2f}")
               .highlight_min(axis=0, color="#D1FAE5")
               .highlight_max(axis=0, color="#FEE2E2"),
            use_container_width=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Inventory Simulation
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    policy = InventoryPolicy(
        reorder_point=rop_slider, order_quantity=eoq_slider,
        lead_time_days=lead_time, holding_cost_per_unit=h_cost,
        stockout_cost_per_unit=s_cost,
        initial_inventory=float(df_f["Inventory_Level"].iloc[0]) if "Inventory_Level" in df_f.columns else avg_d * 3,
    )
    sim = simulate_inventory(demand, policy)
    s_  = sim.summary()

    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, label, value, good_high=True):
        color = GREEN if good_high else RED
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
        </div>""", unsafe_allow_html=True)

    kpi(k1, "Service Level",       f"{s_['service_level_%']}%",    good_high=True)
    kpi(k2, "Stockout Rate",       f"{s_['stockout_rate_%']}%",    good_high=False)
    kpi(k3, "Total Holding Cost",  f"${s_['total_holding_cost_$']:,.0f}", good_high=False)
    kpi(k4, "Total Op. Cost",      f"${s_['total_operating_cost_$']:,.0f}", good_high=False)

    st.markdown("<br>", unsafe_allow_html=True)

    sim_df = pd.DataFrame({"Day": range(len(sim.daily_inventory)),
                           "Inventory": sim.daily_inventory,
                           "Demand":    sim.daily_demand})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["Inventory"],
        name="Inventory Level", fill="tozeroy",
        fillcolor="rgba(37,99,235,0.07)", line=dict(color=BLUE, width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=sim_df["Day"], y=sim_df["Demand"], name="Demand",
        line=dict(color=RED, width=1.2, dash="dash")
    ))
    fig.add_hline(y=rop_slider, line_dash="dot", line_color=AMBER,
                  annotation_text=f"ROP = {rop_slider}", annotation_position="top left")
    fig.update_layout(
        title="Simulated Inventory vs Demand",
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(title="Day", showgrid=False),
        yaxis=dict(title="Units", gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full simulation metrics"):
        st.json(s_)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — What-If Scenarios
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    scenarios = {
        "Conservative": InventoryPolicy(avg_d*4, avg_d*10, lead_time, h_cost, s_cost, avg_d*5),
        "Current Policy": InventoryPolicy(rop_slider, eoq_slider, lead_time, h_cost, s_cost, avg_d*3),
        "Lean":          InventoryPolicy(avg_d*1.5, avg_d*3, lead_time, h_cost, s_cost, avg_d*2),
        "EOQ Optimised": InventoryPolicy(rop, eoq, lead_time, h_cost, s_cost, avg_d*3),
    }
    cmp_df = run_scenario_comparison(demand, scenarios)

    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Holding Cost", x=cmp_df.index,
                             y=cmp_df["total_holding_cost_$"], marker_color=BLUE))
        fig.add_trace(go.Bar(name="Lost Sales Cost", x=cmp_df.index,
                             y=cmp_df["total_lost_sales_cost_$"], marker_color=RED))
        fig.update_layout(
            barmode="stack", title="Total Cost Breakdown",
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=36, b=0),
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F3F4F6", title="Cost ($)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        colors = [GREEN if s >= 95 else AMBER for s in cmp_df["service_level_%"]]
        fig2.add_trace(go.Bar(
            x=cmp_df.index, y=cmp_df["service_level_%"],
            marker_color=colors, marker_line_width=0,
        ))
        fig2.add_hline(y=95, line_dash="dot", line_color=GREY,
                       annotation_text="95% target")
        fig2.update_layout(
            title="Service Level by Scenario",
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#F3F4F6", title="%", range=[0,105]),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Scenario Comparison Table")
    display_cols = ["stockout_rate_%", "service_level_%",
                    "total_holding_cost_$", "total_lost_sales_cost_$",
                    "total_operating_cost_$", "n_orders_placed"]
    st.dataframe(
        cmp_df[display_cols].style
            .format({"stockout_rate_%": "{:.1f}%", "service_level_%": "{:.1f}%",
                     "total_holding_cost_$": "${:,.0f}", "total_lost_sales_cost_$": "${:,.0f}",
                     "total_operating_cost_$": "${:,.0f}", "n_orders_placed": "{:.0f}"})
            .highlight_min(subset=["total_operating_cost_$", "stockout_rate_%"], color="#D1FAE5")
            .highlight_max(subset=["service_level_%"], color="#D1FAE5"),
        use_container_width=True
    )
