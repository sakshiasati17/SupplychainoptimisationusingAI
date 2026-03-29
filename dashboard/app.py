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
BLUE   = "#2563EB"
NAVY   = "#0F172A"
INDIGO = "#6366F1"
GREEN  = "#10B981"
RED    = "#EF4444"
AMBER  = "#F59E0B"
SLATE  = "#64748B"
BG     = "#F8FAFC"
WHITE  = "#FFFFFF"
BORDER = "#E2E8F0"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"]  {{ background: {BG}; }}
[data-testid="stSidebar"]           {{ display: none; }}
[data-testid="collapsedControl"]    {{ display: none; }}
#MainMenu, footer                   {{ visibility: hidden; }}
.block-container {{
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 100% !important;
}}

/* ── Force light toolbar/header bar ── */
[data-testid="stHeader"],
[data-testid="stToolbar"]  {{ background: {WHITE} !important; border-bottom: 1px solid {BORDER}; }}
[data-testid="stDecoration"] {{ display: none !important; }}

/* ── Force light widgets ── */
[data-baseweb="select"] > div {{ background: {WHITE} !important; border-color: {BORDER} !important; }}
[data-baseweb="select"] span  {{ color: {NAVY} !important; }}
[data-baseweb="input"]        {{ background: {WHITE} !important; border-color: {BORDER} !important; }}
[data-baseweb="input"] input  {{ color: {NAVY} !important; background: {WHITE} !important; }}
[data-testid="stSelectbox"]   label,
[data-testid="stNumberInput"] label {{
    color: {SLATE} !important; font-size: 0.75rem !important; font-weight: 600 !important;
}}
[data-testid="stNumberInput"] button {{
    background: {BG} !important; border-color: {BORDER} !important; color: {NAVY} !important;
}}

/* ── KPI cards ── */
.kpi-card {{
    background: {WHITE};
    border: 1px solid {BORDER};
    border-top: 3px solid {BLUE};
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}}
.kpi-card.green {{ border-top-color: {GREEN}; }}
.kpi-card.amber {{ border-top-color: {AMBER}; }}
.kpi-card.red   {{ border-top-color: {RED}; }}
.kpi-label {{
    font-size: 0.67rem; font-weight: 700; color: {SLATE};
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.3rem;
}}
.kpi-value {{
    font-size: 1.75rem; font-weight: 700; color: {NAVY}; line-height: 1.1;
}}
.kpi-sub {{ font-size: 0.72rem; color: {SLATE}; margin-top: 0.25rem; }}

/* ── Stat rows ── */
.stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0; border-bottom: 1px solid {BORDER};
    font-size: 0.82rem;
}}
.stat-label {{ color: {SLATE}; }}
.stat-val   {{ font-weight: 600; color: {NAVY}; }}

/* ── Section divider ── */
.sec-header {{
    font-size: 0.68rem; font-weight: 700; color: {SLATE};
    text-transform: uppercase; letter-spacing: 0.07em;
    padding-bottom: 0.45rem; border-bottom: 1px solid {BORDER};
    margin-bottom: 0.75rem; margin-top: 0.25rem;
}}

/* ── Insight banner ── */
.insight-box {{
    background: #EFF6FF; border: 1px solid #BFDBFE;
    border-radius: 10px; padding: 1rem 1.25rem;
    display: flex; gap: 1rem; align-items: flex-start;
    margin-top: 1rem;
}}

/* ── Control panel ── */
.ctrl-panel {{
    background: {WHITE}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 1.1rem 1.2rem;
}}
.ctrl-header {{
    font-size: 0.68rem; font-weight: 700; color: {SLATE};
    text-transform: uppercase; letter-spacing: 0.07em;
    padding-bottom: 0.5rem; border-bottom: 1px solid {BORDER};
    margin-bottom: 0.8rem;
}}
.rec-box {{
    background: {BG}; border-radius: 8px;
    padding: 0.75rem 0.9rem; margin-top: 1rem; font-size: 0.76rem;
}}

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child button {{
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.1rem !important;
    color: {SLATE} !important;
}}
div[data-testid="stTabs"] > div:first-child button[aria-selected="true"] {{
    color: {BLUE} !important;
    font-weight: 600 !important;
}}
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


# ── Header ────────────────────────────────────────────────────────────────────
hdr_l, hdr_r = st.columns([6, 1])
with hdr_l:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.75rem;
                padding:0.5rem 0 0.25rem 0;
                border-bottom:2px solid {BORDER};margin-bottom:1.4rem;">
        <span style="font-size:1.5rem">📦</span>
        <div>
            <div style="font-size:1.15rem;font-weight:700;color:{NAVY};
                        letter-spacing:-0.02em;line-height:1.1;">
                Supply Chain AI
            </div>
            <div style="font-size:0.72rem;color:{SLATE};margin-top:1px;">
                Demand Forecasting &amp; Inventory Intelligence Platform
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Filter row (native Streamlit widgets — no HTML wrapper) ───────────────────
fc1, fc2, fc3, fc4, fc5 = st.columns([1.4, 1.4, 0.9, 0.9, 0.9])
with fc1:
    store   = st.selectbox("Store", sorted(df["Store_ID"].unique()))
with fc2:
    product = st.selectbox("Product", sorted(df["Product_ID"].unique()))
with fc3:
    lead_time = st.number_input("Lead Time (days)", 1, 14, 3)
with fc4:
    h_cost = st.number_input("Holding $/unit/day", 0.1, 5.0, 0.5, 0.1)
with fc5:
    s_cost = st.number_input("Stockout $/unit lost", 1.0, 20.0, 5.0, 0.5)

st.markdown("<hr style='border:none;border-top:1px solid #E2E8F0;margin:0.5rem 0 1.2rem 0'>",
            unsafe_allow_html=True)


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

def kpi(col, label, value, sub="", accent=""):
    col.markdown(f"""
    <div class="kpi-card {accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Avg Daily Demand",  f"{avg_d:.0f} units", f"σ ±{std_d:.0f}")
kpi(k2, "EOQ",               f"{eoq:.0f} units",   "Optimal order qty",       "green")
kpi(k3, "Safety Stock",      f"{ss:.0f} units",    f"{lead_time}d lead time", "amber")
kpi(k4, "Reorder Point",     f"{rop:.0f} units",   "Trigger restocking")
kpi(k5, "History",           f"{len(ts):,} days",
    f"{ts.index[0].date()} → {ts.index[-1].date()}")

st.markdown("<br>", unsafe_allow_html=True)


# ── Shared chart defaults ─────────────────────────────────────────────────────
def base_layout(title="", height=340):
    return dict(
        title=dict(text=title, font=dict(size=13, color=NAVY), x=0, xanchor="left"),
        height=height,
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        font=dict(family="system-ui, sans-serif", size=11, color=NAVY),
        margin=dict(l=4, r=4, t=48, b=4),
        xaxis=dict(showgrid=False, showline=True, linecolor=BORDER, zeroline=False),
        yaxis=dict(gridcolor="#F1F5F9", showline=False, zeroline=False),
        legend=dict(orientation="h", y=1.18, x=0, font_size=11, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER, font_size=12),
        hovermode="x unified",
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
    col_a, col_b = st.columns([3, 1], gap="large")

    with col_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.values, name="Daily Demand",
            line=dict(color=BLUE, width=1.3),
            fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
        ))
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.rolling(7).mean(), name="7-day MA",
            line=dict(color=AMBER, width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=ts.index, y=ts.rolling(28).mean(), name="28-day MA",
            line=dict(color=GREEN, width=2, dash="dashdot"),
        ))
        fig.update_layout(**base_layout("Daily Units Sold", 350))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-header">Demand Statistics</div>', unsafe_allow_html=True)
        stats = [
            ("Min",    f"{demand.min():.0f} units"),
            ("Max",    f"{demand.max():.0f} units"),
            ("Median", f"{np.median(demand):.0f} units"),
            ("P90",    f"{np.percentile(demand,90):.0f} units"),
            ("CV",     f"{std_d/avg_d*100:.1f}%"),
        ]
        for lbl, val in stats:
            st.markdown(f"""
            <div class="stat-row">
                <span class="stat-label">{lbl}</span>
                <span class="stat-val">{val}</span>
            </div>""", unsafe_allow_html=True)

        if "Category" in df_f.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            cat = df_f.groupby("Category")["Units_Sold"].mean().sort_values()
            clrs = [BLUE if i == len(cat) - 1 else "#BFDBFE" for i in range(len(cat))]
            fig2 = go.Figure(go.Bar(
                x=cat.values, y=cat.index, orientation="h",
                marker_color=clrs, marker_line_width=0,
            ))
            fig2.update_layout(**base_layout("By Category", 240))
            fig2.update_layout(margin=dict(l=4, r=4, t=48, b=4),
                               xaxis=dict(showgrid=True, gridcolor="#F1F5F9"),
                               yaxis=dict(showgrid=False),
                               legend=dict(visible=False))
            st.plotly_chart(fig2, use_container_width=True)

    # Seasonal heatmap
    if len(ts) > 60:
        heat_df = ts.reset_index()
        heat_df.columns = ["Date", "Units_Sold"]
        heat_df["Month"] = heat_df["Date"].dt.strftime("%b")
        heat_df["Year"]  = heat_df["Date"].dt.year
        pivot = heat_df.pivot_table(index="Year", columns="Month",
                                    values="Units_Sold", aggfunc="mean")
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot[[m for m in months if m in pivot.columns]]

        fig3 = go.Figure(go.Heatmap(
            z=pivot.values, x=pivot.columns,
            y=[str(y) for y in pivot.index],
            colorscale=[[0,"#EFF6FF"],[0.5,"#93C5FD"],[1,BLUE]],
            showscale=True, hoverongaps=False,
            colorbar=dict(thickness=10, len=0.8),
        ))
        fig3.update_layout(**base_layout("Seasonal Demand Heatmap", 200))
        fig3.update_layout(margin=dict(l=4, r=60, t=48, b=4),
                           legend=dict(visible=False))
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Forecast Models
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    sl_col, _ = st.columns([1, 3])
    with sl_col:
        horizon = st.slider("Forecast horizon (days)", 7, 90, 30)

    TEST    = min(30, len(ts) // 4)
    train_s = ts.iloc[:-TEST]
    test_s  = ts.iloc[-TEST:]
    actual  = test_s.values
    results = []

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index[-90:], y=ts.values[-90:], name="Actual",
        line=dict(color=NAVY, width=2),
    ))

    # ── Vertical split line (epoch-ms avoids plotly datetime str bug) ──
    split_ms = int(pd.Timestamp(test_s.index[0]).timestamp() * 1000)
    fig.add_shape(
        type="line",
        x0=split_ms, x1=split_ms, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color=BORDER, width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=split_ms, y=1, xref="x", yref="paper",
        text="test window", showarrow=False,
        font=dict(size=10, color=SLATE), yanchor="bottom", xanchor="left",
    )

    # Naive
    naive_pred = np.full(TEST, train_s.iloc[-1])
    results.append(evaluate_all(actual, naive_pred, "Naive Baseline"))
    fig.add_trace(go.Scatter(
        x=test_s.index, y=naive_pred, name="Naive",
        line=dict(color=SLATE, dash="dot", width=1.5),
    ))

    # Moving Average
    ma_pred = np.full(TEST, train_s.rolling(7).mean().iloc[-1])
    results.append(evaluate_all(actual, ma_pred, "Moving Average"))
    fig.add_trace(go.Scatter(
        x=test_s.index, y=ma_pred, name="Moving Avg",
        line=dict(color=AMBER, dash="dash", width=1.5),
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
                FCOLS = [c for c in models["features"] if c in feat_df.columns]
                if len(feat_df) > TEST and FCOLS:
                    xgb_pred = models["xgboost"].predict(feat_df.tail(TEST)[FCOLS])
                    results.append(evaluate_all(actual[:len(xgb_pred)], xgb_pred, "XGBoost"))
                    fig.add_trace(go.Scatter(
                        x=test_s.index[:len(xgb_pred)], y=xgb_pred,
                        name="XGBoost", line=dict(color=GREEN, width=2),
                    ))
                    lgb_pred = models["lightgbm"].predict(feat_df.tail(TEST)[FCOLS])
                    results.append(evaluate_all(actual[:len(lgb_pred)], lgb_pred, "LightGBM"))
                    fig.add_trace(go.Scatter(
                        x=test_s.index[:len(lgb_pred)], y=lgb_pred,
                        name="LightGBM", line=dict(color=INDIGO, width=2),
                    ))
        except Exception:
            pass

    # LSTM + uncertainty band
    if "lstm" in models:
        try:
            lstm_pred = models["lstm"].predict(train_s, steps=TEST)
            results.append(evaluate_all(actual[:len(lstm_pred)], lstm_pred, "LSTM"))
            std_band  = np.std(actual[:len(lstm_pred)] - lstm_pred)
            idx = list(test_s.index[:len(lstm_pred)])
            fig.add_trace(go.Scatter(
                x=idx + idx[::-1],
                y=list(lstm_pred + std_band) + list((lstm_pred - std_band)[::-1]),
                fill="toself", fillcolor="rgba(239,68,68,0.08)",
                line=dict(width=0), name="LSTM ±1σ", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=test_s.index[:len(lstm_pred)], y=lstm_pred,
                name="LSTM", line=dict(color=RED, width=2),
            ))
        except Exception:
            pass

    fig.update_layout(**base_layout(f"Model Comparison — {TEST}-day Test Window", 400))
    st.plotly_chart(fig, use_container_width=True)

    if results:
        st.markdown('<div class="sec-header">Model Performance Metrics</div>',
                    unsafe_allow_html=True)
        cmp = compare_models(results)
        def _hl(s):
            return ["background:#D1FAE5;color:#065F46;font-weight:600"
                    if v else "" for v in (s == s.min())]
        st.dataframe(
            cmp.style.apply(_hl, axis=0).format("{:.2f}"),
            use_container_width=True, height=180,
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Inventory Simulation
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    ctrl_col, chart_col = st.columns([1, 3], gap="large")

    with ctrl_col:
        st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)
        st.markdown('<div class="ctrl-header">Policy Controls</div>',
                    unsafe_allow_html=True)
        rop_slider = st.slider("Reorder Point (units)", 10, int(avg_d * 8), int(rop))
        eoq_slider = st.slider("Order Quantity (units)", 50, int(avg_d * 20), int(eoq))
        st.markdown(f"""
        <div class="rec-box">
            <div style="font-size:0.72rem;font-weight:700;color:{NAVY};margin-bottom:0.4rem;">
                AI Recommendation
            </div>
            <div style="color:{SLATE};line-height:2;font-size:0.75rem;">
                EOQ &nbsp;<b style="color:{BLUE}">{eoq:.0f}</b> units<br>
                Safety Stock &nbsp;<b style="color:{AMBER}">{ss:.0f}</b> units<br>
                Reorder at &nbsp;<b style="color:{GREEN}">{rop:.0f}</b> units
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

    with chart_col:
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
        sl  = float(s_["service_level_%"])
        sor = float(s_["stockout_rate_%"])

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f"""<div class="kpi-card {'green' if sl>=95 else 'amber'}">
            <div class="kpi-label">Service Level</div>
            <div class="kpi-value" style="color:{'#10B981' if sl>=95 else '#F59E0B'}">{sl:.1f}%</div>
            <div class="kpi-sub">{'On target ✓' if sl>=95 else 'Below 95%'}</div>
        </div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="kpi-card {'red' if sor>2 else 'green'}">
            <div class="kpi-label">Stockout Rate</div>
            <div class="kpi-value" style="color:{'#EF4444' if sor>2 else '#10B981'}">{sor:.1f}%</div>
            <div class="kpi-sub">{'Needs attention' if sor>2 else 'Within limit'}</div>
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
            name="Inventory", fill="tozeroy",
            fillcolor="rgba(37,99,235,0.06)", line=dict(color=BLUE, width=1.8),
        ))
        fig.add_trace(go.Scatter(
            x=sim_df["Day"], y=sim_df["Demand"],
            name="Demand", line=dict(color=RED, width=1.2, dash="dash"),
        ))
        fig.add_hline(y=rop_slider, line_dash="dot", line_color=AMBER, line_width=1.5)
        fig.add_annotation(
            x=0, y=rop_slider, xref="x", yref="y",
            text=f"  ROP = {rop_slider}", showarrow=False,
            font=dict(size=10, color=AMBER), xanchor="left",
        )
        fig.update_layout(**base_layout("Simulated Inventory Level vs Demand", 300))
        fig.update_layout(xaxis=dict(title="Day"))
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

    cl, cr = st.columns(2, gap="large")

    SCENARIO_COLORS = [SLATE, BLUE, AMBER, GREEN]

    def hex_rgba(h, alpha):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    with cl:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Holding Cost", x=cmp_df.index,
            y=cmp_df["total_holding_cost_$"],
            marker_color=[hex_rgba(c, 0.85) for c in SCENARIO_COLORS],
            marker_line_width=0,
        ))
        fig.add_trace(go.Bar(
            name="Lost Sales", x=cmp_df.index,
            y=cmp_df["total_lost_sales_cost_$"],
            marker_color=[hex_rgba(c, 0.35) for c in SCENARIO_COLORS],
            marker_line_width=0,
        ))
        fig.update_layout(**base_layout("Cost Breakdown by Scenario", 300))
        fig.update_layout(barmode="stack",
                          yaxis=dict(title="Cost ($)", gridcolor="#F1F5F9"))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        colors = [GREEN if s >= 95 else AMBER for s in cmp_df["service_level_%"]]
        fig2 = go.Figure(go.Bar(
            x=cmp_df.index, y=cmp_df["service_level_%"],
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in cmp_df["service_level_%"]],
            textposition="outside",
        ))
        fig2.add_hline(y=95, line_dash="dot", line_color=SLATE, line_width=1.5)
        fig2.add_annotation(
            x=len(cmp_df) - 0.5, y=95, xref="x", yref="y",
            text="95% SLA", showarrow=False,
            font=dict(size=10, color=SLATE), yanchor="bottom",
        )
        fig2.update_layout(**base_layout("Service Level by Scenario", 300))
        fig2.update_layout(yaxis=dict(title="Service Level (%)",
                                      range=[0, 112], gridcolor="#F1F5F9"))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-header">Scenario Comparison</div>',
                unsafe_allow_html=True)

    display_cols = [
        "service_level_%", "stockout_rate_%",
        "total_holding_cost_$", "total_lost_sales_cost_$",
        "total_operating_cost_$", "n_orders_placed",
    ]
    rename = {
        "service_level_%":        "Service Level (%)",
        "stockout_rate_%":        "Stockout Rate (%)",
        "total_holding_cost_$":   "Holding Cost ($)",
        "total_lost_sales_cost_$":"Lost Sales ($)",
        "total_operating_cost_$": "Total Op. Cost ($)",
        "n_orders_placed":        "Orders Placed",
    }
    st.dataframe(
        cmp_df[display_cols].rename(columns=rename)
        .style.format({
            "Service Level (%)":  "{:.1f}%",
            "Stockout Rate (%)":  "{:.1f}%",
            "Holding Cost ($)":   "${:,.0f}",
            "Lost Sales ($)":     "${:,.0f}",
            "Total Op. Cost ($)": "${:,.0f}",
            "Orders Placed":      "{:.0f}",
        })
        .highlight_min(subset=["Total Op. Cost ($)", "Stockout Rate (%)"], color="#D1FAE5")
        .highlight_max(subset=["Service Level (%)"], color="#D1FAE5"),
        use_container_width=True, height=195,
    )

    best      = cmp_df["total_operating_cost_$"].idxmin()
    best_cost = cmp_df.loc[best, "total_operating_cost_$"]
    savings   = cmp_df["total_operating_cost_$"].max() - best_cost
    svc       = cmp_df.loc[best, "service_level_%"]

    st.markdown(f"""
    <div class="insight-box">
        <span style="font-size:1.3rem">💡</span>
        <div>
            <div style="font-size:0.83rem;font-weight:700;color:{NAVY}">
                Recommendation: <span style="color:{BLUE}">{best}</span>
            </div>
            <div style="font-size:0.77rem;color:{SLATE};margin-top:0.2rem;line-height:1.6">
                Lowest total cost at <b>${best_cost:,.0f}</b> —
                saves <b>${savings:,.0f}</b> vs worst case.
                Delivers <b>{svc:.1f}%</b> service level.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;padding:2.5rem 0 1rem;
            font-size:0.72rem;color:{SLATE};">
    Supply Chain AI &nbsp;·&nbsp; Demand Forecasting &amp; Inventory Intelligence
    &nbsp;·&nbsp; Built by <b>Sakshi Asati</b>
</div>""", unsafe_allow_html=True)
