"""
app.py  –  Tele-Law Access Equity Analysis  |  Streamlit Dashboard
===================================================================
Professional interactive UI with:
  • Sidebar slicers  (state, district, year, caste filter)
  • Dynamic KPI cards
  • Plotly charts  (pie, bar, line, heatmap, scatter, funnel)
  • IBM Granite AI insight panel
  • Data-table explorer with download

Run: streamlit run app.py
"""

import sys
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Tele-Law Equity Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── local imports ─────────────────────────────────────────────────────────────
from main import _generate_demo_data
from src.data_ingestion import clean_and_standardise, load_and_prepare
from src.analysis import (
    gender_disparity, caste_disparity, regional_disparity,
    trend_analysis, compute_equity_score, correlation_analysis,
)
from src.granite_ai import run_granite_insights

# ─────────────────────────────────────────────────────────────────────────────
# THEME & PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_BLUE   = "#3b82d4"
C_PURPLE = "#7c5cd8"
C_AMBER  = "#f59e0b"
C_GREEN  = "#22c55e"
C_RED    = "#ef4444"
C_TEAL   = "#06b6d4"
C_SLATE  = "#64748b"

CASTE_PALETTE  = [C_BLUE, C_PURPLE, C_AMBER, C_GREEN]
GENDER_PALETTE = [C_BLUE, C_AMBER, C_TEAL]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Segoe UI, sans-serif", size=12, color="#1f2328"),
    margin=dict(l=10, r=10, t=40, b=10),
)

# Default horizontal legend — spread as **H_LEGEND where a top legend is wanted
H_LEGEND = dict(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General ── */
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
}
/* ── Hide default Streamlit chrome ── */
#MainMenu, footer {visibility: hidden;}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1f2328 !important;
}
section[data-testid="stSidebar"] * {color: #e5e7eb !important;}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label {
    color: #9ca3af !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: .06em;
}

/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
    transition: box-shadow .2s;
}
.kpi-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,.12); }
.kpi-value {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
}
.kpi-label {
    font-size: 11px;
    color: #57606a;
    text-transform: uppercase;
    letter-spacing: .05em;
    margin-top: 4px;
}
.kpi-delta {
    font-size: 12px;
    margin-top: 6px;
    font-weight: 600;
}
.delta-up   { color: #22c55e; }
.delta-down { color: #ef4444; }
.delta-neu  { color: #64748b; }

/* ── Section headers ── */
.section-header {
    font-size: 16px;
    font-weight: 700;
    color: #1f2328;
    border-left: 4px solid #3b82d4;
    padding-left: 10px;
    margin: 8px 0 16px;
}

/* ── Insight box ── */
.insight-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-left: 4px solid #3b82d4;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 13px;
    line-height: 1.7;
    color: #374151;
}

/* ── Badge pills ── */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 4px;
}
.badge-blue   { background:#eff6ff; color:#3b82d4; border:1px solid #bfdbfe; }
.badge-green  { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; }
.badge-red    { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.badge-purple { background:#f5f3ff; color:#7c3aed; border:1px solid #ddd6fe; }

/* ── Divider ── */
hr.thin { border: none; border-top: 1px solid #e5e7eb; margin: 16px 0; }

/* ── Metrics override ── */
[data-testid="stMetricValue"] { font-size: 26px !important; }

/* ── Chart containers ── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 6px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Generating demo dataset …")
def _load_demo() -> pd.DataFrame:
    raw = _generate_demo_data()
    return clean_and_standardise(raw)

@st.cache_data(show_spinner="Loading CSV …")
def _load_csv(path: str) -> pd.DataFrame:
    return load_and_prepare(path)

def get_dataframe(source: str, uploaded=None, csv_path: str = "") -> pd.DataFrame:
    if source == "Demo Data":
        return _load_demo()
    elif source == "Upload CSV" and uploaded is not None:
        import io
        raw = pd.read_csv(io.BytesIO(uploaded.read()))
        return clean_and_standardise(raw)
    elif source == "Local CSV" and csv_path:
        return _load_csv(csv_path)
    return _load_demo()


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def kpi_html(value: str, label: str, color: str, delta: str = "", delta_type: str = "neu") -> str:
    delta_html = (
        f'<div class="kpi-delta delta-{delta_type}">{delta}</div>' if delta else ""
    )
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""

def section_header(text: str) -> None:
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)

def apply_layout(fig) -> None:
    fig.update_layout(**PLOTLY_LAYOUT)

def equity_badge(score: float) -> str:
    if score >= 70:
        return f'<span class="badge badge-green">High Equity {score:.0f}</span>'
    elif score >= 45:
        return f'<span class="badge badge-blue">Moderate {score:.0f}</span>'
    else:
        return f'<span class="badge badge-red">Low Equity {score:.0f}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  – Slicers
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px;">
        <div style="font-size:22px;">⚖️</div>
        <div style="font-size:14px;font-weight:700;color:#fff;">Tele-Law Equity</div>
        <div style="font-size:11px;color:#6b7280;margin-top:2px;">IBM Cloud Lite · IBM Granite AI</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Data source selector
    st.markdown('<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Data Source</div>', unsafe_allow_html=True)
    data_source = st.selectbox(
        "Data Source", ["Demo Data", "Upload CSV", "Local CSV"],
        label_visibility="collapsed"
    )
    uploaded_file = None
    csv_path_input = ""
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Tele-Law CSV", type=["csv"], label_visibility="collapsed")
    elif data_source == "Local CSV":
        csv_path_input = st.text_input("CSV Path", value="data/raw/telelaw_data.csv", label_visibility="collapsed")

    st.markdown("---")

    # Load data
    with st.spinner("Loading data …"):
        df_full = get_dataframe(data_source, uploaded_file, csv_path_input)

    all_states = sorted(df_full["state_name"].unique().tolist())
    all_years  = sorted(df_full["financial_year"].unique().tolist())

    # ── Slicers ───────────────────────────────────────────────────────────────
    st.markdown('<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;">Financial Year</div>', unsafe_allow_html=True)
    sel_years = st.multiselect("Years", all_years, default=all_years, label_visibility="collapsed")

    st.markdown('<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:10px;margin-bottom:4px;">State / UT</div>', unsafe_allow_html=True)
    sel_states = st.multiselect("States", all_states, default=all_states, label_visibility="collapsed")

    # District depends on state selection
    avail_districts = sorted(
        df_full[df_full["state_name"].isin(sel_states or all_states)]["district_name"].unique().tolist()
    )
    st.markdown('<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-top:10px;margin-bottom:4px;">District</div>', unsafe_allow_html=True)
    sel_districts = st.multiselect("Districts", avail_districts, default=[], label_visibility="collapsed",
                                   placeholder="All districts")

    st.markdown("---")

    st.markdown('<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Equity Settings</div>', unsafe_allow_html=True)
    equity_threshold = st.slider("Marginalized Threshold (%)", 50, 90, 70, 5)
    female_threshold = st.slider("Female Participation Floor (%)", 15, 50, 30, 5)

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#9ca3af;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">IBM Granite AI</div>', unsafe_allow_html=True)
    run_ai = st.toggle("Generate AI Insights", value=False)
    if run_ai:
        st.caption("Requires IBM_CLOUD_API_KEY and WATSONX_PROJECT_ID in .env")

    st.markdown("---")
    st.markdown('<div style="font-size:10px;color:#4b5563;text-align:center;">IBM Cloud Lite · watsonx.ai<br/>data.gov.in · FY 2021-22 to 2024-25</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────

years_f     = sel_years  if sel_years  else all_years
states_f    = sel_states if sel_states else all_states
districts_f = sel_districts if sel_districts else avail_districts

df = df_full[
    df_full["financial_year"].isin(years_f) &
    df_full["state_name"].isin(states_f) &
    df_full["district_name"].isin(districts_f)
].copy()

# Re-apply dynamic equity threshold
df["meets_equity_threshold"] = df["marginalized_pct"] >= equity_threshold

if df.empty:
    st.warning("No data matches the current filters. Please broaden your selection.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# RUN ANALYSIS  (cached by filter hash)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_analysis(df_hash: str, _df: pd.DataFrame) -> dict:
    from src.analysis import run_full_analysis
    return run_full_analysis(_df)

df_hash = f"{len(df)}_{df['total_cases_registered'].sum()}_{equity_threshold}"
results = run_analysis(df_hash, df)

g_res  = results["gender"]
c_res  = results["caste"]
r_res  = results["regional"]
t_res  = results["trend"]
eq_df  = results["equity"]
corr   = results["correlation"]

g_nat  = g_res["national"]
c_nat  = c_res["national"]
yearly = t_res["yearly"]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="background:linear-gradient(135deg,#1f2328 0%,#2d3748 100%);
            border-radius:12px;padding:24px 32px;margin-bottom:24px;">
    <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div style="font-size:36px;">⚖️</div>
        <div>
            <div style="font-size:22px;font-weight:700;color:#fff;">
                Tele-Law Access Equity Dashboard
            </div>
            <div style="font-size:13px;color:#9ca3af;margin-top:4px;">
                Problem Statement 37 · District-wise Case Registration &amp; Advice Data · FY 2021-22 to 2024-25
            </div>
            <div style="margin-top:10px;">
                <span style="background:#3b82d4;color:#fff;font-size:11px;padding:3px 10px;
                             border-radius:20px;margin-right:6px;">IBM Cloud Lite</span>
                <span style="background:#7c5cd8;color:#fff;font-size:11px;padding:3px 10px;
                             border-radius:20px;margin-right:6px;">IBM Granite AI</span>
                <span style="background:#22c55e;color:#fff;font-size:11px;padding:3px 10px;
                             border-radius:20px;">data.gov.in</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Active filter summary
filter_parts = []
if sel_years  and len(sel_years)  < len(all_years):  filter_parts.append(f"**Years:** {', '.join(sel_years)}")
if sel_states and len(sel_states) < len(all_states):  filter_parts.append(f"**States:** {len(sel_states)} selected")
if sel_districts: filter_parts.append(f"**Districts:** {len(sel_districts)} selected")
if filter_parts:
    st.info("🔍 Active filters — " + "  ·  ".join(filter_parts), icon=None)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION TABS
# ─────────────────────────────────────────────────────────────────────────────

tab_overview, tab_gender, tab_caste, tab_regional, tab_equity, tab_trend, tab_ai, tab_data = st.tabs([
    "📊 Overview", "👥 Gender", "🏷️ Caste", "🗺️ Regional",
    "🏆 Equity Index", "📈 Trends", "🤖 AI Insights", "🗃️ Data Explorer"
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:

    # ── KPI Row 1 ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    kpis = [
        (c1, f"{g_nat['total_cases']:,}", "Total Cases",    C_BLUE,   "", "neu"),
        (c2, f"{g_nat['female_pct']}%",   "Female Share",   C_AMBER,
             ("▲ Above 30%" if g_nat["female_pct"] >= 30 else "▼ Below 30%"),
             "up" if g_nat["female_pct"] >= 30 else "down"),
        (c3, f"{c_nat['sc_pct']}%",       "SC Share",       C_PURPLE, "", "neu"),
        (c4, f"{c_nat['st_pct']}%",       "ST Share",       C_TEAL,   "", "neu"),
        (c5, f"{c_nat['obc_pct']}%",      "OBC Share",      C_GREEN,  "", "neu"),
        (c6, f"{c_nat['marginalized_pct']}%", "Marginalized",C_RED,
             ("▲ Meets threshold" if c_nat["marginalized_pct"] >= equity_threshold else "▼ Below threshold"),
             "up" if c_nat["marginalized_pct"] >= equity_threshold else "down"),
        (c7, f"{c_res['gini_coefficient']}", "Gini Index",  C_SLATE,  "", "neu"),
    ]
    for col, val, lbl, clr, delta, dtype in kpis:
        with col:
            st.markdown(kpi_html(val, lbl, clr, delta, dtype), unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── KPI Row 2 – latest year stats ──────────────────────────────────────
    if not yearly.empty:
        latest = yearly.iloc[-1]
        prev   = yearly.iloc[-2] if len(yearly) > 1 else None
        c1, c2, c3, c4 = st.columns(4)
        advice_delta = ""
        advice_type  = "neu"
        if prev is not None:
            diff = round(latest["advice_rate_pct"] - prev["advice_rate_pct"], 1)
            advice_delta = f"{'▲' if diff >= 0 else '▼'} {abs(diff):.1f}pp vs prior year"
            advice_type  = "up" if diff >= 0 else "down"

        with c1:
            st.markdown(kpi_html(
                f"{int(latest['total_cases']):,}", "Cases (Latest Year)", C_BLUE, "", "neu"
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi_html(
                f"{latest['advice_rate_pct']:.1f}%", "Advice Rate", C_GREEN,
                advice_delta, advice_type
            ), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi_html(
                f"{latest['female_pct']:.1f}%", "Female % (Latest)", C_AMBER, "", "neu"
            ), unsafe_allow_html=True)
        with c4:
            eq_top = eq_df.iloc[0] if len(eq_df) > 0 else None
            if eq_top is not None:
                st.markdown(kpi_html(
                    f"{eq_top['equity_score']:.0f}/100",
                    f"Top Equity: {eq_top['district_name']}",
                    C_GREEN, eq_top["state_name"], "up"
                ), unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Overview Charts ────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        section_header("Gender Distribution")
        fig = go.Figure(go.Pie(
            labels=["Male", "Female", "Other"],
            values=[g_nat["male_pct"], g_nat["female_pct"], g_nat["other_pct"]],
            marker_colors=GENDER_PALETTE,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, title_text="Gender Share (National)", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Caste-wise Distribution")
        fig2 = go.Figure(go.Pie(
            labels=["SC", "ST", "OBC", "General"],
            values=[c_nat["sc_pct"], c_nat["st_pct"], c_nat["obc_pct"], c_nat["general_pct"]],
            marker_colors=CASTE_PALETTE,
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title_text="Caste Share (National)", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Top States Bar ─────────────────────────────────────────────────────
    section_header("State-wise Total Cases Registered")
    ss = r_res["state_summary"].head(15).reset_index()
    fig3 = px.bar(
        ss, x="state_name", y="total_cases",
        color="advice_rate_pct",
        color_continuous_scale=["#ef4444", "#f59e0b", "#22c55e"],
        range_color=[50, 100],
        labels={"state_name": "State", "total_cases": "Total Cases",
                "advice_rate_pct": "Advice Rate %"},
        hover_data=["total_districts", "avg_female_pct", "avg_marginalized_pct"],
    )
    fig3.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_tickangle=-35,
                       coloraxis_colorbar=dict(title="Advice<br>Rate %", len=0.7))
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – GENDER
# ═══════════════════════════════════════════════════════════════════════════
with tab_gender:
    section_header("Gender-wise Case Registration Analysis")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_html(f"{g_nat['male_pct']}%", "Male Beneficiaries", C_BLUE), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html(f"{g_nat['female_pct']}%", "Female Beneficiaries", C_AMBER), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_html(f"{g_nat['other_pct']}%", "Other Gender", C_TEAL), unsafe_allow_html=True)
    with k4:
        skewed_n = len(g_res["skewed_districts"])
        st.markdown(kpi_html(str(skewed_n), f"Districts < {female_threshold}% Female", C_RED), unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Stacked bar: gender by state
        by_state = g_res["by_state"].reset_index().head(20)
        fig = go.Figure()
        for gender, color, label in [
            ("male_pct", C_BLUE, "Male"),
            ("female_pct", C_AMBER, "Female"),
            ("other_pct", C_TEAL, "Other"),
        ]:
            fig.add_trace(go.Bar(
                name=label, x=by_state["state_name"], y=by_state[gender],
                marker_color=color,
                hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
            ))
        fig.update_layout(**PLOTLY_LAYOUT, **H_LEGEND, barmode="stack", height=380,
                          title_text="Gender Share by State (%)",
                          xaxis_tickangle=-35,
                          yaxis_title="Percentage (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Line: female % trend by year
        by_year = g_res["by_year"].reset_index()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=by_year["financial_year"], y=by_year["female_pct"],
            mode="lines+markers+text", name="Female %",
            line=dict(color=C_AMBER, width=3),
            marker=dict(size=10),
            text=[f"{v:.1f}%" for v in by_year["female_pct"]],
            textposition="top center",
        ))
        fig2.add_trace(go.Scatter(
            x=by_year["financial_year"], y=by_year["male_pct"],
            mode="lines+markers", name="Male %",
            line=dict(color=C_BLUE, width=3),
            marker=dict(size=10),
        ))
        fig2.add_hline(y=female_threshold, line_dash="dash", line_color=C_RED,
                       annotation_text=f"{female_threshold}% Floor", annotation_position="bottom right")
        fig2.update_layout(**PLOTLY_LAYOUT, **H_LEGEND, height=380,
                           title_text="Gender Share Trend (Year-over-Year)",
                           yaxis_title="Percentage (%)")
        st.plotly_chart(fig2, use_container_width=True)

    # Skewed districts table
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    section_header(f"Districts with Female Participation Below {female_threshold}%")
    skewed = g_res["skewed_districts"][g_res["skewed_districts"]["female_pct"] < female_threshold]
    if len(skewed):
        fig3 = px.scatter(
            skewed, x="female_pct", y="total_cases_registered",
            color="state_name", size="total_cases_registered",
            hover_data=["district_name", "financial_year"],
            labels={"female_pct": "Female %", "total_cases_registered": "Total Cases",
                    "state_name": "State"},
            title=f"Skewed Districts — Female % vs Case Volume",
        )
        fig3.add_vline(x=female_threshold, line_dash="dash", line_color=C_RED)
        fig3.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(
            skewed.rename(columns={"state_name":"State","district_name":"District",
                                   "financial_year":"Year","female_pct":"Female %",
                                   "total_cases_registered":"Cases"}),
            use_container_width=True, height=220, hide_index=True,
        )
    else:
        st.success(f"No districts fall below the {female_threshold}% female threshold with current filters.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – CASTE
# ═══════════════════════════════════════════════════════════════════════════
with tab_caste:
    section_header("Caste-wise & Marginalization Analysis")

    k1, k2, k3, k4, k5 = st.columns(5)
    for col, val, lbl, clr in [
        (k1, f"{c_nat['sc_pct']}%",          "SC Share",           C_BLUE),
        (k2, f"{c_nat['st_pct']}%",           "ST Share",           C_PURPLE),
        (k3, f"{c_nat['obc_pct']}%",          "OBC Share",          C_AMBER),
        (k4, f"{c_nat['marginalized_pct']}%", "Total Marginalized", C_GREEN),
        (k5, str(c_res["gini_coefficient"]),  "Gini Coefficient",   C_RED),
    ]:
        with col:
            st.markdown(kpi_html(val, lbl, clr), unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Stacked bar by state
        cs = c_res["by_state"].reset_index().head(15)
        fig = go.Figure()
        for cat, clr, lbl in [
            ("sc_pct", C_BLUE,"SC"),("st_pct", C_PURPLE,"ST"),
            ("obc_pct", C_AMBER,"OBC"),("general_pct", C_GREEN,"General"),
        ]:
            fig.add_trace(go.Bar(name=lbl, x=cs["state_name"], y=cs[cat],
                                 marker_color=clr))
        fig.update_layout(**PLOTLY_LAYOUT, **H_LEGEND, barmode="stack", height=380,
                          title_text="Caste Distribution by State (%)",
                          xaxis_tickangle=-35, yaxis_title="%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Trend: marginalized %
        cy = c_res["by_year"].reset_index()
        fig2 = go.Figure()
        for cat, clr, lbl in [
            ("sc_pct", C_BLUE,"SC"),("st_pct", C_PURPLE,"ST"),("obc_pct", C_AMBER,"OBC")
        ]:
            fig2.add_trace(go.Scatter(x=cy["financial_year"], y=cy[cat],
                                      mode="lines+markers", name=lbl,
                                      line=dict(color=clr, width=2.5),
                                      marker=dict(size=8)))
        fig2.update_layout(**PLOTLY_LAYOUT, **H_LEGEND, height=380,
                           title_text="Caste Share Trend (Year-over-Year)",
                           yaxis_title="%")
        st.plotly_chart(fig2, use_container_width=True)

    # Heatmap: marginalized % by state × year
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    section_header("Marginalized Group Heatmap (State × Year)")
    pivot = (
        df.groupby(["state_name","financial_year"])["marginalized_pct"]
          .mean().unstack("financial_year").fillna(0)
    )
    fig3 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        text=np.round(pivot.values,1),
        texttemplate="%{text}%",
        colorbar=dict(title="Marg. %"),
        hovertemplate="State: %{y}<br>Year: %{x}<br>Marginalized: %{z:.1f}%<extra></extra>",
    ))
    fig3.update_layout(**PLOTLY_LAYOUT,
                       height=max(350, len(pivot)*28),
                       title_text="Marginalized % — State × Financial Year",
                       xaxis_title="Financial Year", yaxis_title="State")
    st.plotly_chart(fig3, use_container_width=True)

    # Underserved districts
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    section_header(f"Underserved Districts (Marginalized % < {equity_threshold}%)")
    und = c_res["underserved_districts"]
    if len(und):
        fig4 = px.histogram(
            und, x="marginalized_pct", color="state_name", nbins=20,
            labels={"marginalized_pct": "Marginalized %","state_name":"State"},
            title=f"Distribution of Districts Below {equity_threshold}% Marginalized Threshold",
        )
        fig4.add_vline(x=equity_threshold, line_dash="dash", line_color=C_RED,
                       annotation_text=f"Threshold {equity_threshold}%")
        fig4.update_layout(**PLOTLY_LAYOUT, height=320)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.success("All districts meet the equity threshold with current filters.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – REGIONAL
# ═══════════════════════════════════════════════════════════════════════════
with tab_regional:
    section_header("Geographic & Regional Disparity Analysis")

    ss = r_res["state_summary"].reset_index()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(kpi_html(str(len(ss)), "States / UTs", C_BLUE), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html(f"{ss['avg_cases_per_csc'].mean():.1f}", "Avg Cases / CSC", C_PURPLE), unsafe_allow_html=True)
    with k3:
        top_advice = ss.nlargest(1,"advice_rate_pct")
        top_name   = top_advice.iloc[0]["state_name"] if len(top_advice) else "N/A"
        top_rate   = top_advice.iloc[0]["advice_rate_pct"] if len(top_advice) else 0
        st.markdown(kpi_html(f"{top_rate:.1f}%", f"Best Advice Rate: {top_name}", C_GREEN), unsafe_allow_html=True)
    with k4:
        n_out = len(r_res["outlier_districts"])
        st.markdown(kpi_html(str(n_out), "Outlier Districts", C_RED), unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Bubble chart: cases vs advice rate, size = CSCs
        fig = px.scatter(
            ss, x="total_cases", y="advice_rate_pct",
            size="total_cscs", color="avg_marginalized_pct",
            text="state_name",
            color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
            range_color=[30, 100],
            labels={"total_cases":"Total Cases","advice_rate_pct":"Advice Rate %",
                    "total_cscs":"Total CSCs","avg_marginalized_pct":"Marg. %"},
            title="State Bubble Chart — Cases vs Advice Rate (size = CSCs)",
        )
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.add_hline(y=80, line_dash="dot", line_color=C_GREEN,
                      annotation_text="80% Target", annotation_position="bottom right")
        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Funnel: advice rate ranking
        funnel_df = ss.sort_values("advice_rate_pct", ascending=True).head(12)
        clrs = [C_GREEN if v >= 80 else C_AMBER if v >= 60 else C_RED
                for v in funnel_df["advice_rate_pct"]]
        fig2 = go.Figure(go.Bar(
            x=funnel_df["advice_rate_pct"], y=funnel_df["state_name"],
            orientation="h", marker_color=clrs,
            text=[f"{v:.1f}%" for v in funnel_df["advice_rate_pct"]],
            textposition="outside",
        ))
        fig2.add_vline(x=80, line_dash="dash", line_color=C_GREEN)
        fig2.update_layout(**PLOTLY_LAYOUT, height=420,
                           title_text="Advice Delivery Rate by State",
                           xaxis_title="Advice Rate (%)", yaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    # Top / Bottom districts
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    section_header("District-level Access Density (Cases per CSC)")

    col3, col4 = st.columns(2)
    for col, data, title, clr in [
        (col3, r_res["district_top20"], "Top 20 Districts", C_GREEN),
        (col4, r_res["district_bot20"], "Bottom 20 Districts", C_RED),
    ]:
        with col:
            fig_ = px.bar(
                data.sort_values("cases_per_csc"), x="cases_per_csc",
                y=data["district_name"] + " (" + data["state_name"] + ")",
                orientation="h", color_discrete_sequence=[clr],
                labels={"x":"Cases / CSC","y":"District"},
                title=title,
            )
            fig_.update_layout(**PLOTLY_LAYOUT, height=460, yaxis_title="")
            col.plotly_chart(fig_, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 – EQUITY INDEX
# ═══════════════════════════════════════════════════════════════════════════
with tab_equity:
    section_header("Composite Equity Index — District Scorecard")

    st.markdown("""
    <div class="insight-box">
    <b>How the Equity Index is calculated:</b>
    The index (0–100) is a weighted composite of four normalised indicators:
    <b>Marginalized group share (35%)</b> · <b>Female participation (25%)</b> ·
    <b>Advice delivery rate (25%)</b> · <b>CSC access density (15%)</b>.
    Higher score = more inclusive access.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    avg_eq = eq_df["equity_score"].mean()
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(kpi_html(f"{avg_eq:.1f}", "National Avg Equity Score", C_BLUE), unsafe_allow_html=True)
    with k2: st.markdown(kpi_html(str((eq_df["equity_score"]>=70).sum()), "High-Equity Districts (≥70)", C_GREEN), unsafe_allow_html=True)
    with k3: st.markdown(kpi_html(str((eq_df["equity_score"]<45).sum()), "Low-Equity Districts (<45)", C_RED), unsafe_allow_html=True)
    with k4: st.markdown(kpi_html(str(len(eq_df)), "Total Districts Scored", C_PURPLE), unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Scatter: equity vs marginalized %
        fig = px.scatter(
            eq_df, x="marginalized_pct", y="female_pct",
            color="equity_score", size="total_cases",
            color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
            hover_name="district_name",
            hover_data={"state_name":True,"equity_score":True,"advice_rate_pct":True},
            labels={"marginalized_pct":"Marginalized %","female_pct":"Female %",
                    "equity_score":"Equity Score","total_cases":"Total Cases"},
            title="District Equity Space — Marginalized % vs Female % (color = Equity Score)",
        )
        fig.add_hline(y=female_threshold, line_dash="dash", line_color=C_RED, opacity=0.5)
        fig.add_vline(x=equity_threshold, line_dash="dash", line_color=C_PURPLE, opacity=0.5)
        fig.update_layout(**PLOTLY_LAYOUT, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Top 15 and bottom 15 bar
        n_show = min(15, len(eq_df) // 2)
        top_eq = eq_df.head(n_show).copy()
        bot_eq = eq_df.tail(n_show).copy()
        combined = pd.concat([top_eq.assign(group="Top"), bot_eq.assign(group="Bottom")])
        fig2 = px.bar(
            combined.sort_values("equity_score"),
            x="equity_score",
            y=combined["district_name"] + "\n(" + combined["state_name"] + ")",
            color="group",
            color_discrete_map={"Top": C_GREEN, "Bottom": C_RED},
            orientation="h",
            title=f"Top {n_show} vs Bottom {n_show} Districts",
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=420, yaxis_title="",
                           xaxis_title="Equity Score (0–100)",
                           legend=dict(title="Group"))
        st.plotly_chart(fig2, use_container_width=True)

    # State-level equity heatmap
    st.markdown("<hr class='thin'>", unsafe_allow_html=True)
    section_header("State-level Equity Scorecard")
    state_eq = eq_df.groupby("state_name")["equity_score"].mean().round(1).reset_index()
    state_eq.columns = ["State","Avg Equity Score"]
    state_eq = state_eq.sort_values("Avg Equity Score", ascending=False)
    state_eq["Rating"] = state_eq["Avg Equity Score"].apply(
        lambda x: "🟢 High" if x >= 70 else ("🟡 Moderate" if x >= 45 else "🔴 Low")
    )
    fig3 = px.bar(
        state_eq, x="State", y="Avg Equity Score",
        color="Avg Equity Score",
        color_continuous_scale=["#ef4444","#f59e0b","#22c55e"],
        range_color=[20, 90],
        text="Avg Equity Score",
        title="Average Equity Score by State",
    )
    fig3.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig3.update_layout(**PLOTLY_LAYOUT, height=360, xaxis_tickangle=-35)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("View full equity scorecard table"):
        st.dataframe(
            eq_df[["district_name","state_name","equity_score","marginalized_pct",
                   "female_pct","advice_rate_pct","cases_per_csc","total_cases"]]
              .rename(columns={"district_name":"District","state_name":"State",
                               "equity_score":"Equity Score","marginalized_pct":"Marg. %",
                               "female_pct":"Female %","advice_rate_pct":"Advice Rate %",
                               "cases_per_csc":"Cases/CSC","total_cases":"Total Cases"}),
            use_container_width=True, height=400, hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 – TRENDS
# ═══════════════════════════════════════════════════════════════════════════
with tab_trend:
    section_header("Year-over-Year Trend Analysis")

    if yearly.empty:
        st.info("No trend data available for current filters.")
    else:
        yr = yearly.reset_index()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            growth = yr["cases_yoy_growth_pct"].dropna().iloc[-1] if len(yr) > 1 else 0
            st.markdown(kpi_html(
                f"{'+'if growth>=0 else ''}{growth:.1f}%", "Case Growth (YoY)", C_BLUE,
                delta=yr["financial_year"].iloc[-1], delta_type="up" if growth>=0 else "down"
            ), unsafe_allow_html=True)
        with k2:
            advice_g = yr["advice_yoy_growth_pct"].dropna().iloc[-1] if len(yr)>1 else 0
            st.markdown(kpi_html(
                f"{'+'if advice_g>=0 else ''}{advice_g:.1f}%", "Advice Growth (YoY)", C_GREEN,
                delta_type="up" if advice_g>=0 else "down"
            ), unsafe_allow_html=True)
        with k3:
            st.markdown(kpi_html(
                f"{yr['female_pct'].iloc[-1]:.1f}%", "Female % (Latest)", C_AMBER
            ), unsafe_allow_html=True)
        with k4:
            st.markdown(kpi_html(
                f"{yr['marginalized_pct'].iloc[-1]:.1f}%", "Marginalized % (Latest)", C_PURPLE
            ), unsafe_allow_html=True)

        st.markdown("<hr class='thin'>", unsafe_allow_html=True)

        # Dual-axis: cases bars + rate lines
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=yr["financial_year"], y=yr["total_cases"]/1000,
            name="Total Cases (K)", marker_color=C_BLUE, opacity=0.7,
        ), secondary_y=False)
        for metric, clr, lbl in [
            ("advice_rate_pct", C_GREEN, "Advice Rate %"),
            ("marginalized_pct", C_AMBER, "Marginalized %"),
            ("female_pct", C_RED, "Female %"),
        ]:
            fig.add_trace(go.Scatter(
                x=yr["financial_year"], y=yr[metric],
                mode="lines+markers+text",
                name=lbl, line=dict(color=clr, width=2.5),
                marker=dict(size=9),
                text=[f"{v:.1f}%" for v in yr[metric]],
                textposition="top center",
            ), secondary_y=True)

        fig.update_yaxes(title_text="Total Cases (Thousands)", secondary_y=False)
        fig.update_yaxes(title_text="Rate / Percentage (%)", secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, **H_LEGEND, height=420,
                          title_text="Year-over-Year Tele-Law Metrics",
                          xaxis_title="Financial Year")
        st.plotly_chart(fig, use_container_width=True)

        # Growth rate waterfall
        st.markdown("<hr class='thin'>", unsafe_allow_html=True)
        section_header("Year-over-Year Growth Rates")
        growth_df = yr.dropna(subset=["cases_yoy_growth_pct"])
        if len(growth_df):
            fig2 = go.Figure()
            colors = [C_GREEN if v >= 0 else C_RED for v in growth_df["cases_yoy_growth_pct"]]
            fig2.add_trace(go.Bar(
                x=growth_df["financial_year"],
                y=growth_df["cases_yoy_growth_pct"],
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in growth_df["cases_yoy_growth_pct"]],
                textposition="outside",
                name="Case Growth %",
            ))
            fig2.add_hline(y=0, line_color="#e5e7eb", line_width=1)
            fig2.update_layout(**PLOTLY_LAYOUT, height=300,
                               title_text="Case Registration Growth Rate (%)",
                               yaxis_title="Growth (%)")
            st.plotly_chart(fig2, use_container_width=True)

        # Trend table
        with st.expander("View yearly metrics table"):
            display_yr = yr.copy()
            display_yr.columns = [c.replace("_"," ").title() for c in display_yr.columns]
            st.dataframe(display_yr, use_container_width=True, hide_index=True)

        # Correlation heatmap
        st.markdown("<hr class='thin'>", unsafe_allow_html=True)
        section_header("Pearson Correlation Matrix — Equity Indicators")
        fig3 = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values,2),
            texttemplate="%{text}",
            colorbar=dict(title="r"),
            hovertemplate="X: %{x}<br>Y: %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        fig3.update_layout(**PLOTLY_LAYOUT, height=450,
                           title_text="Pearson Correlation — Key Equity Metrics")
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 – AI INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_ai:
    section_header("IBM Granite AI — Policy Insights")

    st.markdown("""
    <div class="insight-box" style="margin-bottom:16px;">
    <b>IBM Granite AI</b> (<code>ibm/granite-13b-instruct-v2</code>) generates policy-grade narratives
    from the analysis results via IBM watsonx.ai (IBM Cloud Lite).
    Configure <code>IBM_CLOUD_API_KEY</code> and <code>WATSONX_PROJECT_ID</code> in <code>.env</code>
    then toggle <b>Generate AI Insights</b> in the sidebar.
    </div>
    """, unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        _raw_key = st.secrets.get("IBM_CLOUD_API_KEY", os.getenv("IBM_CLOUD_API_KEY", ""))
        api_key_ok = bool(str(_raw_key).replace("your_ibm_cloud_api_key_here", "").strip())
        st.markdown(
            f'<span class="badge {"badge-green" if api_key_ok else "badge-red"}">IBM_CLOUD_API_KEY: {"✓ Set" if api_key_ok else "✗ Not set"}</span>',
            unsafe_allow_html=True
        )
    with col_cfg2:
        _raw_proj = st.secrets.get("WATSONX_PROJECT_ID", os.getenv("WATSONX_PROJECT_ID", ""))
        proj_ok = bool(str(_raw_proj).replace("your_watsonx_project_id_here", "").strip())
        st.markdown(
            f'<span class="badge {"badge-green" if proj_ok else "badge-red"}">WATSONX_PROJECT_ID: {"✓ Set" if proj_ok else "✗ Not set"}</span>',
            unsafe_allow_html=True
        )
    with col_cfg3:
        model = st.secrets.get("GRANITE_MODEL_ID", os.getenv("GRANITE_MODEL_ID", "ibm/granite-13b-instruct-v2"))
        st.markdown(f'<span class="badge badge-blue">Model: {model}</span>', unsafe_allow_html=True)

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    if run_ai:
        if not (api_key_ok and proj_ok):
            st.warning("IBM credentials not configured. Please update your `.env` file.")
        else:
            with st.spinner("Contacting IBM Granite via watsonx.ai …"):
                insights = run_granite_insights(results)
            st.success("IBM Granite AI insights generated successfully.")
    else:
        insights = {
            "executive_summary": None,
            "gender_insight":    None,
            "caste_insight":     None,
            "regional_insight":  None,
        }

    def insight_section(title: str, icon: str, key: str) -> None:
        st.markdown(f"**{icon} {title}**")
        if insights.get(key):
            st.markdown(f'<div class="insight-box">{insights[key]}</div>', unsafe_allow_html=True)
        else:
            placeholder = {
                "executive_summary": (
                    f"National Tele-Law data shows {g_nat['female_pct']}% female participation and "
                    f"{c_nat['marginalized_pct']}% marginalized group share. "
                    f"A Gini coefficient of {c_res['gini_coefficient']} indicates "
                    f"{'low' if c_res['gini_coefficient'] < 0.2 else 'moderate'} inequality. "
                    f"Enable IBM Granite in the sidebar for full AI-generated policy recommendations."
                ),
                "gender_insight": (
                    f"Female participation stands at {g_nat['female_pct']}% nationally. "
                    f"{len(g_res['skewed_districts'])} districts have female share below 30%. "
                    f"Enable IBM Granite AI for targeted intervention recommendations."
                ),
                "caste_insight": (
                    f"SC ({c_nat['sc_pct']}%), ST ({c_nat['st_pct']}%), OBC ({c_nat['obc_pct']}%) "
                    f"together represent {c_nat['marginalized_pct']}% of cases. "
                    f"{len(c_res['underserved_districts'])} districts fall below the {equity_threshold}% equity threshold. "
                    f"Enable IBM Granite AI for in-depth policy analysis."
                ),
                "regional_insight": (
                    f"Regional disparity is evident across {len(r_res['state_summary'])} states. "
                    f"Top districts register up to {r_res['district_top20']['cases_per_csc'].max():.1f} cases/CSC "
                    f"vs {r_res['district_bot20']['cases_per_csc'].min():.1f} in the lowest. "
                    f"Enable IBM Granite AI for geographic policy recommendations."
                ),
            }
            st.markdown(
                f'<div class="insight-box" style="opacity:0.75;">'
                f'<em>📝 Analytical summary (enable Granite for AI-generated insights):</em><br><br>'
                f'{placeholder.get(key, "")}</div>',
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)

    insight_section("Executive Summary", "📋", "executive_summary")
    insight_section("Gender Equity Analysis", "👥", "gender_insight")
    insight_section("Caste & Marginalization Analysis", "🏷️", "caste_insight")
    insight_section("Regional Disparity Analysis", "🗺️", "regional_insight")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 – DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════
with tab_data:
    section_header("Data Explorer — Filtered Dataset")

    # Row-level stats
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Rows in view",        f"{len(df):,}")
    with k2: st.metric("States",              df["state_name"].nunique())
    with k3: st.metric("Districts",           df["district_name"].nunique())
    with k4: st.metric("Financial Years",     df["financial_year"].nunique())

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # Column picker
    all_cols = df.columns.tolist()
    default_cols = ["state_name","district_name","financial_year","total_cscs",
                    "total_cases_registered","cases_advised","female_pct",
                    "marginalized_pct","advice_rate_pct","cases_per_csc","equity_flag"]
    default_cols = [c for c in default_cols if c in all_cols]

    col_pick, col_sort = st.columns([3, 1])
    with col_pick:
        sel_cols = st.multiselect("Select columns to display", all_cols, default=default_cols)
    with col_sort:
        sort_col = st.selectbox("Sort by", ["total_cases_registered","advice_rate_pct",
                                             "marginalized_pct","female_pct","cases_per_csc"],
                                 index=0)

    if sel_cols:
        display_df = df[sel_cols].sort_values(
            sort_col if sort_col in df.columns else "total_cases_registered",
            ascending=False
        )
        st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)

        # Download
        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download filtered data as CSV",
            data=csv_bytes,
            file_name="telelaw_filtered.csv",
            mime="text/csv",
        )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # Summary statistics
    section_header("Descriptive Statistics")
    num_df = df.select_dtypes(include="number")
    st.dataframe(num_df.describe().round(2).T, use_container_width=True, height=360)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style="border:none;border-top:1px solid #e5e7eb;margin:32px 0 12px;">
<div style="text-align:center;font-size:12px;color:#9ca3af;padding-bottom:16px;">
    ⚖️ Tele-Law Access Equity Dashboard &nbsp;·&nbsp;
    Problem Statement 37 &nbsp;·&nbsp;
    IBM Cloud Lite &nbsp;·&nbsp; IBM Granite AI &nbsp;·&nbsp;
    data.gov.in · FY 2021-22 to 2024-25 &nbsp;·&nbsp;
    Made with IBM Bob
</div>
""", unsafe_allow_html=True)
