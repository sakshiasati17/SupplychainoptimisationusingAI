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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain AI",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour palette ────────────────────────────────────────────────────────────
BLUE    = "#2563EB"
NAVY    = "#0F172A"
INDIGO  = "#6366F1"
GREEN   = "#10B981"
RED     = "#EF4444"
AMBER   = "#F59E0B"
SLATE   = "#64748B"
BG      = "#F8FAFC"
WHITE   = "#FFFFFF"
BORDER  = "#E2E8F0"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Reset & base ── */
[data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stSidebar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none; }}
#MainMenu, footer {{ visibility: hidden; }}
.block-container {{ padding: 0 2rem 2rem 2rem !important; max-width: 100% !important; }}

/* ── Typography ── */
h1, h2, h3, h4 {{ color: {NAVY}; }}

/* ── Top header bar ── */
.header-bar {{
    background: {WHITE};
    border-bottom: 1px solid {BORDER};
    padding: 0.85rem 2rem;
    margin: 0 -2rem 1.5rem -2rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}}
.header-logo {{
    width: 32px; height: 32px;
    background: {BLUE};
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 1rem;
}}
.header-title {{
    font-size: 1.05rem; font-weight: 700;
    color: {NAVY}; letter-spacing: -0.01em;
}}
.header-sub {{
    font-size: 0.75rem; color: {SLATE}; margin-top: 1px;
}}

/* ── Filter row ── */
.filter-bar {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.85rem 1.25rem;
    margin-bottom: 1.25rem;
}}

/* ── KPI cards ── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.25rem;
}}
.kpi-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    position: relative;
    overflow: hidden;
}}
.kpi-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: {BLUE};
    border-radius: 10px 10px 0 0;
}}
.kpi-card.green::before {{ background: {GREEN}; }}
.kpi-card.red::before   {{ background: {RED}; }}
.kpi-card.amber::before {{ background: {AMBER}; }}
.kpi-label {{
    font-size: 0.68rem; font-weight: 600;
    color: {SLATE}; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 0.35rem;
}}
.kpi-value {{
    font-size: 1.65rem; font-weight: 700;
    color: {NAVY}; line-height: 1;
}}
.kpi-sub {{
    font-size: 0.72rem; color: {SLATE}; margin-top: 0.3rem;
}}

/* ── Chart card ── */
.chart-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1.2rem 1.4rem 0.6rem 1.4rem;
    margin-bottom: 1rem;
}}
.chart-title {{
    font-size: 0.82rem; font-weight: 600;
    color: {NAVY}; margin-bottom: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}}

/* ── Section header ── */
.section-label {{
    font-size: 0.7rem; font-weight: 700;
    color: {SLATE}; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 0.6rem;
    padding-bottom: 0.4rem; border-bottom: 1px solid {BORDER};
}}

/* ── Sim control card ── */
.ctrl-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
    height: 100%;
}}
.ctrl-title {{
    font-size: 0.7rem; font-weight: 700;
    color: {SLATE}; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 0.75rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
}}

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 10px 10px 0 0;
    padding: 0 0.5rem;
    border-bottom: none;
    margin-bottom: -1px;
}}
div[data-testid="stTabs"] button {{
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: {SLATE} !important;
    border-radius: 0 !important;
    padding: 0.65rem 1rem !important;
}}
div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {BLUE} !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {BLUE} !important;
}}
div[data-testid="stTabsContent"] {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-radius: 0 10px 10px 10px;
    padding: 1.5rem;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{ border-radius: 8px; overflow: hidden; }}

/* ── Scenario badge ── */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
}}
.badge-green {{ background: #D1FAE5; color: #065F46; }}
.badge-blue  {{ background: #DBEAFE; color: #1D4ED8; }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
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

df     = load_data()
models = load_models()


# ── Header bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <div class="header-logo">📦</div>
  <div>
    <div class="header-title">Supply Chain AI</div>
    <div class="header-sub">Demand Forecasting &amp; Inventory Intelligence Platform</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Filter row ────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([1.2, 1.2, 1, 1, 1.2, 1.2])
    with fc1:
        stores  = sorted(df["Store_ID"].unique())
        store   = st.selectbox("Store", stores, label_visibility="visible")
    with fc2:
        products = sorted(df["Product_ID"].unique())
        product  = st.selectbox("Product", products, label_visibility="visible")
    with fc3:
        lead_time = st.number_input("Lead Time (days)", 1, 14, 3)
    with fc4:
        h_cost = st.number_input("Holding $/unit/day", 0.1, 5.0, 0.5, 0.1)
    with fc5:
        s_cost = st.number_input("Stockout $/unit lost", 1.0, 20.0, 5.0, 0.5)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Filtered data ─────────────────────────────────────────────────────────────
mask = (df["Store_ID"] == store) & (df["Product_ID"] == product)
df_f = df[mask].sort_values("Date") if mask.any() else df.sort_values("Date")
ts   = df_f.groupby("Date")["Units_Sold"].sum().sort_index()
demand = ts.values

avg_d = demand.mean()
std_d = demand.std()
eoq   = compute_eoq(avg_d * 365)
ss    = compute_safety_stock(avg_d, std_d, lead_time)
rop   = compute_reorder_point(avg_d, lead_time, ss)


# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi_card(col, label, value, sub="", accent="blue"):
    col.markdown(f"""
    <div class="kpi-card {accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi_card(k1, "Avg Daily Demand",  f"{avg_d:.0f}",      f"Std dev ±{std_d:.0f} units")
kpi_card(k2, "EOQ",               f"{eoq:.0f}",         "Optimal order qty",        "green")
kpi_card(k3, "Safety Stock",      f"{ss:.0f}",          f"At {lead_time}d lead time","amber")
kpi_card(k4, "Reorder Point",     f"{rop:.0f}",         "Trigger new order at",     "blue")
kpi_card(k5, "History",           f"{len(ts):,} days",  f"{ts.index[0].date()} → {ts.index[-1].date()}")

st.markdown("<br>", unsafe_allow_html=True)


# ── Chart helpers ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    font=dict(family="Inter, system-ui, sans-serif", size=11, color=NAVY),
    margin=dict(l=8, r=8, t=40, b=8),
    xaxis=dict(showgrid=False, showline=True, linecolor=BORDER, zeroline=False),
    yaxis=dict(gridcolor="#F1F5F9", showline=False, zeroline=False),
    legend=dict(orientation="h", y=1.14, x=0, font_size=11),
    hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER, font_size=12),
)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Demand Analysis  ",
    "  Forecast Models  ",
    "  Inventory Simulation  ",
    "  Scenario Planning  ",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Demand Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_a, col_b = st.columns([3, 1], gap="medium")

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.values, name="Daily Demand",
            line=dict(color=BLUE, width=1.5),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>%{y:.0f} units<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.rolling(7).mean(), name="7-day MA",
            line=dict(color=AMBER, width=2.5, dash="dot"),
            hovertemplate="%{y:.0f}<extra>7-day MA</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.rolling(28).mean(), name="28-day MA",
            line=dict(color=GREEN, width=2, dash="dashdot"),
            hovertemplate="%{y:.0f}<extra>28-day MA</extra>"
        ))
        fig.update_layout(**CHART_LAYOUT, title="Daily Units Sold", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if "Category" in df_f.columns:
            cat = df_f.groupby("Category")["Units_Sold"].mean().sort_values()
            clrs = [BLUE if i == len(cat)-1 else "#BFDBFE" for i in range(len(cat))]
            fig2 = go.Figure(go.Bar(
                x=cat.values, y=cat.index, orientation="h",
                marker_color=clrs, marker_line_width=0,
                hovertemplate="%{y}: %{x:.0f} units<extra></extra>"
            ))
            fig2.update_layout(**CHART_LAYOUT, title="Avg Demand by Category", height=340)
            fig2.update_layout(xaxis=dict(showgrid=True, gridcolor="#F1F5F9"), yaxis=dict(showgrid=False))
            st.plotly_chart(fig2, use_container_width=True)

    # Second row: heatmap + stats
    col_c, col_d = st.columns([2, 1], gap="medium")

    with col_c:
        if len(ts) > 60:
            heat_df = ts.reset_index()
            heat_df.columns = ["Date", "Units_Sold"]
            heat_df["Month"] = heat_df["Date"].dt.strftime("%b")
            heat_df["Year"]  = heat_df["Date"].dt.year
            pivot = heat_df.pivot_table(index="Year", columns="Month", values="Units_Sold", aggfunc="mean")
            month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            pivot = pivot[[m for m in month_order if m in pivot.columns]]

            fig3 = go.Figure(go.Heatmap(
                z=pivot.values, x=pivot.columns, y=[str(y) for y in pivot.index],
                colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1, BLUE]],
                showscale=True, hoverongaps=False,
                hovertemplate="<b>%{x} %{y}</b><br>Avg %{z:.0f} units<extra></extra>",
                colorbar=dict(thickness=10, len=0.8)
            ))
            fig3.update_layout(**CHART_LAYOUT, title="Seasonal Demand Heatmap", height=220)
            fig3.update_layout(margin=dict(l=8, r=60, t=40, b=8))
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-label">Demand Statistics</div>', unsafe_allow_html=True)
        stats = {
            "Min":    f"{demand.min():.0f} units",
            "Max":    f"{demand.max():.0f} units",
            "Median": f"{np.median(demand):.0f} units",
            "CV":     f"{(std_d/avg_d*100):.1f}%",
            "P90":    f"{np.percentile(demand,90):.0f} units",
        }
        for label, val in stats.items():
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;
                        padding:0.45rem 0;border-bottom:1px solid {BORDER};
                        font-size:0.82rem;'>
                <span style='color:{SLATE}'>{label}</span>
                <span style='font-weight:600;color:{NAVY}'>{val}</span>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast Models
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    ctrl_col, _ = st.columns([1, 3])
    with ctrl_col:
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

    TEST    = min(30, len(ts) // 4)
    train_s = ts.iloc[:-TEST]
    test_s  = ts.iloc[-TEST:]
    actual  = test_s.values
    results = []

    fig = go.Figure()
    # Context window — last 90 days of actuals
    fig.add_trace(go.Scatter(
        x=ts.index[-90:], y=ts.values[-90:], name="Actual",
        line=dict(color=NAVY, width=2),
        hovertemplate="%{x|%b %d}: %{y:.0f}<extra>Actual</extra>"
    ))
    # Train/test split line
    fig.add_vline(x=str(test_s.index[0]), line_dash="dash",
                  line_color=BORDER, line_width=1.5,
                  annotation_text="Test window", annotation_font_size=10,
                  annotation_font_color=SLATE)

    # Naive
    naive_pred = np.full(TEST, train_s.iloc[-1])
    results.append(evaluate_all(actual, naive_pred, "Naive Baseline"))
    fig.add_trace(go.Scatter(
        x=test_s.index, y=naive_pred, name="Naive",
        line=dict(color=SLATE, dash="dot", width=1.5),
        hovertemplate="%{y:.0f}<extra>Naive</extra>"
    ))

    # Moving Average
    ma_pred = np.full(TEST, train_s.rolling(7).mean().iloc[-1])
    results.append(evaluate_all(actual, ma_pred, "Moving Average"))
    fig.add_trace(go.Scatter(
        x=test_s.index, y=ma_pred, name="Moving Avg",
        line=dict(color=AMBER, dash="dash", width=1.5),
        hovertemplate="%{y:.0f}<extra>Moving Avg</extra>"
    ))

    # XGBoost + LightGBM
    if "xgboost" in models and "features" in models:
        try:
            feat_path = Path("data/processed/feature_matrix.csv")
            if feat_path.exists():
                feat_df = pd.read_csv(feat_path, parse_dates=["Date"])
                feat_df = feat_df[
                    (feat_df["Store_ID"] == store) &
                    (feat_df["Product_ID"] == product)
                ].sort_values("Date")
                FCOLS = models["features"]
                FCOLS_avail = [c for c in FCOLS if c in feat_df.columns]
                if len(feat_df) > TEST and FCOLS_avail:
                    xgb_pred = models["xgboost"].predict(feat_df.tail(TEST)[FCOLS_avail])
                    results.append(evaluate_all(actual[:len(xgb_pred)], xgb_pred, "XGBoost"))
                    fig.add_trace(go.Scatter(
                        x=test_s.index[:len(xgb_pred)], y=xgb_pred,
                        name="XGBoost", line=dict(color=GREEN, width=2),
                        hovertemplate="%{y:.0f}<extra>XGBoost</extra>"
                    ))
                    lgb_pred = models["lightgbm"].predict(feat_df.tail(TEST)[FCOLS_avail])
                    results.append(evaluate_all(actual[:len(lgb_pred)], lgb_pred, "LightGBM"))
                    fig.add_trace(go.Scatter(
                        x=test_s.index[:len(lgb_pred)], y=lgb_pred,
                        name="LightGBM", line=dict(color=INDIGO, width=2),
                        hovertemplate="%{y:.0f}<extra>LightGBM</extra>"
                    ))
        except Exception:
            pass

    # LSTM + uncertainty band
    if "lstm" in models:
        try:
            lstm_pred = models["lstm"].predict(train_s, steps=TEST)
            results.append(evaluate_all(actual[:len(lstm_pred)], lstm_pred, "LSTM"))
            std_band = np.std(actual[:len(lstm_pred)] - lstm_pred)
            fig.add_trace(go.Scatter(
                x=list(test_s.index[:len(lstm_pred)]) + list(reversed(test_s.index[:len(lstm_pred)])),
                y=list(lstm_pred + std_band) + list(reversed(lstm_pred - std_band)),
                fill="toself", fillcolor="rgba(239,68,68,0.08)",
                line=dict(width=0), showlegend=True, name="LSTM ±1σ",
                hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=test_s.index[:len(lstm_pred)], y=lstm_pred,
                name="LSTM", line=dict(color=RED, width=2),
                hovertemplate="%{y:.0f}<extra>LSTM</extra>"
            ))
        except Exception:
            pass

    fig.update_layout(
        **CHART_LAYOUT,
        title=f"Model Comparison — {TEST}-day Test Window",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model comparison table
    if results:
        st.markdown('<div class="section-label" style="margin-top:0.5rem">Model Performance Metrics</div>',
                    unsafe_allow_html=True)
        cmp = compare_models(results)

        def highlight(s):
            is_min = s == s.min()
            return ["background-color:#D1FAE5;color:#065F46;font-weight:600" if v
                    else "" for v in is_min]

        st.dataframe(
            cmp.style.apply(highlight, axis=0).format("{:.2f}"),
            use_container_width=True,
            height=180,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Inventory Simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    ctrl, charts = st.columns([1, 3], gap="medium")

    with ctrl:
        st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
        st.markdown('<div class="ctrl-title">Policy Controls</div>', unsafe_allow_html=True)
        rop_slider = st.slider("Reorder Point", 10, int(avg_d * 8), int(rop), key="rop")
        eoq_slider = st.slider("Order Quantity", 50, int(avg_d * 20), int(eoq), key="eoq")
        st.markdown(f"""
        <div style='margin-top:1rem;padding:0.75rem;background:{BG};
                    border-radius:8px;font-size:0.75rem;'>
            <div style='font-weight:600;color:{NAVY};margin-bottom:0.4rem'>
                Recommended Policy
            </div>
            <div style='color:{SLATE};line-height:1.8'>
                EOQ: <b style='color:{BLUE}'>{eoq:.0f} units</b><br>
                Safety Stock: <b style='color:{AMBER}'>{ss:.0f} units</b><br>
                Reorder at: <b style='color:{GREEN}'>{rop:.0f} units</b>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with charts:
        policy = InventoryPolicy(
            reorder_point=rop_slider,
            order_quantity=eoq_slider,
            lead_time_days=lead_time,
            holding_cost_per_unit=h_cost,
            stockout_cost_per_unit=s_cost,
            initial_inventory=float(df_f["Inventory_Level"].iloc[0])
                if "Inventory_Level" in df_f.columns else avg_d * 3,
        )
        sim = simulate_inventory(demand, policy)
        s_  = sim.summary()

        # KPI row
        m1, m2, m3, m4 = st.columns(4)
        sl  = float(s_["service_level_%"])
        sor = float(s_["stockout_rate_%"])

        m1.markdown(f"""<div class="kpi-card {'green' if sl >= 95 else 'amber'}">
            <div class="kpi-label">Service Level</div>
            <div class="kpi-value" style="color:{'#10B981' if sl>=95 else '#F59E0B'}">{sl:.1f}%</div>
            <div class="kpi-sub">{'On target ✓' if sl>=95 else 'Below 95% target'}</div>
        </div>""", unsafe_allow_html=True)

        m2.markdown(f"""<div class="kpi-card {'red' if sor > 2 else 'green'}">
            <div class="kpi-label">Stockout Rate</div>
            <div class="kpi-value" style="color:{'#EF4444' if sor>2 else '#10B981'}">{sor:.1f}%</div>
            <div class="kpi-sub">{'Action needed' if sor>2 else 'Within threshold'}</div>
        </div>""", unsafe_allow_html=True)

        m3.markdown(f"""<div class="kpi-card amber">
            <div class="kpi-label">Holding Cost</div>
            <div class="kpi-value">${s_['total_holding_cost_$']:,.0f}</div>
            <div class="kpi-sub">{s_['n_orders_placed']:.0f} orders placed</div>
        </div>""", unsafe_allow_html=True)

        m4.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Total Op. Cost</div>
            <div class="kpi-value">${s_['total_operating_cost_$']:,.0f}</div>
            <div class="kpi-sub">Holding + lost sales</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        sim_df = pd.DataFrame({
            "Day":       range(len(sim.daily_inventory)),
            "Inventory": sim.daily_inventory,
            "Demand":    sim.daily_demand,
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sim_df["Day"], y=sim_df["Inventory"],
            name="Inventory Level",
            fill="tozeroy", fillcolor="rgba(37,99,235,0.06)",
            line=dict(color=BLUE, width=1.8),
            hovertemplate="Day %{x}: %{y:.0f} units<extra>Inventory</extra>"
        ))
        fig.add_trace(go.Scatter(
            x=sim_df["Day"], y=sim_df["Demand"],
            name="Demand", line=dict(color=RED, width=1.2, dash="dash"),
            hovertemplate="Day %{x}: %{y:.0f} units<extra>Demand</extra>"
        ))
        fig.add_hrect(y0=0, y1=rop_slider, fillcolor="rgba(245,158,11,0.05)",
                      line_width=0, annotation_text="")
        fig.add_hline(
            y=rop_slider, line_dash="dot", line_color=AMBER, line_width=1.5,
            annotation_text=f"ROP = {rop_slider}",
            annotation_font_size=11, annotation_font_color=AMBER,
        )
        fig.update_layout(
            **CHART_LAYOUT,
            title="Simulated Inventory Level vs Demand",
            height=310,
        )
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Scenario Planning
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    scenarios = {
        "Conservative": InventoryPolicy(avg_d*4,   avg_d*10, lead_time, h_cost, s_cost, avg_d*5),
        "Current":      InventoryPolicy(rop_slider, eoq_slider, lead_time, h_cost, s_cost, avg_d*3),
        "Lean":         InventoryPolicy(avg_d*1.5,  avg_d*3,  lead_time, h_cost, s_cost, avg_d*2),
        "EOQ Optimal":  InventoryPolicy(rop,        eoq,      lead_time, h_cost, s_cost, avg_d*3),
    }
    cmp_df = run_scenario_comparison(demand, scenarios)

    chart_l, chart_r = st.columns(2, gap="medium")

    with chart_l:
        SCENARIO_COLORS = [SLATE, BLUE, AMBER, GREEN]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Holding Cost", x=cmp_df.index,
            y=cmp_df["total_holding_cost_$"],
            marker_color=[f"{c}CC" for c in SCENARIO_COLORS],
            marker_line_width=0,
            hovertemplate="%{x}: $%{y:,.0f}<extra>Holding</extra>"
        ))
        fig.add_trace(go.Bar(
            name="Lost Sales", x=cmp_df.index,
            y=cmp_df["total_lost_sales_cost_$"],
            marker_color=[f"{c}55" for c in SCENARIO_COLORS],
            marker_line_width=0,
            hovertemplate="%{x}: $%{y:,.0f}<extra>Lost Sales</extra>"
        ))
        fig.update_layout(
            **CHART_LAYOUT,
            barmode="stack", title="Cost Breakdown by Scenario",
            height=300,
        )
        fig.update_layout(yaxis=dict(title="Total Cost ($)", gridcolor="#F1F5F9"))
        st.plotly_chart(fig, use_container_width=True)

    with chart_r:
        colors = [GREEN if s >= 95 else AMBER for s in cmp_df["service_level_%"]]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=cmp_df.index, y=cmp_df["service_level_%"],
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in cmp_df["service_level_%"]],
            textposition="outside", textfont=dict(size=11, color=NAVY),
            hovertemplate="%{x}: %{y:.1f}%<extra>Service Level</extra>"
        ))
        fig2.add_hline(y=95, line_dash="dot", line_color=SLATE, line_width=1.5,
                       annotation_text="95% SLA", annotation_font_size=10,
                       annotation_font_color=SLATE)
        fig2.update_layout(
            **CHART_LAYOUT,
            title="Service Level by Scenario",
            height=300,
        )
        fig2.update_layout(yaxis=dict(title="Service Level (%)", range=[0, 110], gridcolor="#F1F5F9"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-label" style="margin-top:0.25rem">Scenario Comparison</div>',
                unsafe_allow_html=True)

    display_cols = [
        "service_level_%", "stockout_rate_%",
        "total_holding_cost_$", "total_lost_sales_cost_$",
        "total_operating_cost_$", "n_orders_placed",
    ]
    col_labels = {
        "service_level_%":        "Service Level (%)",
        "stockout_rate_%":        "Stockout Rate (%)",
        "total_holding_cost_$":   "Holding Cost ($)",
        "total_lost_sales_cost_$":"Lost Sales ($)",
        "total_operating_cost_$": "Total Op. Cost ($)",
        "n_orders_placed":        "Orders Placed",
    }
    styled = (
        cmp_df[display_cols]
        .rename(columns=col_labels)
        .style
        .format({
            "Service Level (%)":  "{:.1f}%",
            "Stockout Rate (%)":  "{:.1f}%",
            "Holding Cost ($)":   "${:,.0f}",
            "Lost Sales ($)":     "${:,.0f}",
            "Total Op. Cost ($)": "${:,.0f}",
            "Orders Placed":      "{:.0f}",
        })
        .highlight_min(subset=["Total Op. Cost ($)", "Stockout Rate (%)"], color="#D1FAE5")
        .highlight_max(subset=["Service Level (%)"], color="#D1FAE5")
        .set_table_styles([
            {"selector": "th", "props": [
                ("background", BG), ("color", SLATE),
                ("font-size", "0.72rem"), ("text-transform", "uppercase"),
                ("letter-spacing", "0.04em"), ("font-weight", "600"),
                ("padding", "0.6rem 0.8rem"),
            ]},
            {"selector": "td", "props": [
                ("padding", "0.55rem 0.8rem"), ("font-size", "0.84rem"),
            ]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=200)

    # Insight banner — best scenario
    best = cmp_df["total_operating_cost_$"].idxmin()
    best_cost = cmp_df.loc[best, "total_operating_cost_$"]
    worst_cost = cmp_df["total_operating_cost_$"].max()
    savings = worst_cost - best_cost

    st.markdown(f"""
    <div style='margin-top:1rem;padding:1rem 1.25rem;
                background:#EFF6FF;border:1px solid #BFDBFE;
                border-radius:10px;display:flex;align-items:center;gap:1rem;'>
        <div style='font-size:1.4rem'>💡</div>
        <div>
            <div style='font-size:0.82rem;font-weight:700;color:{NAVY}'>
                Recommendation: <span style='color:{BLUE}'>{best}</span>
            </div>
            <div style='font-size:0.78rem;color:{SLATE};margin-top:0.2rem'>
                Best total operating cost at <b>${best_cost:,.0f}</b>
                — saves <b>${savings:,.0f}</b> vs worst scenario.
                Achieves {cmp_df.loc[best, "service_level_%"]:.1f}% service level.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='text-align:center;padding:2rem 0 1rem;
            font-size:0.72rem;color:{SLATE};border-top:1px solid {BORDER};
            margin-top:2rem;'>
    Supply Chain AI · Demand Forecasting &amp; Inventory Intelligence ·
    Built by <b>Sakshi Asati</b>
</div>""", unsafe_allow_html=True)
