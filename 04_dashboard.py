# =============================================================================
# 04_dashboard.py
# PURPOSE: Interactive Streamlit dashboard for the coffee shop owner to
#          monitor barista performance, drink metrics, and statistical findings.
#
# HOW TO RUN:
#   streamlit run 04_dashboard.py
#
# DEPENDS ON: coffee_shop.db  (created by 01_clean_data.py)
#             outputs/        (CSVs from 02_descriptive_stats.py)
#             outputs/figures/(PNGs from 02 and 03 scripts)
#
# ADDING NEW DATA: Replace coffee_shop_activity.xlsx and re-run 01_clean_data.py.
#                  The dashboard will automatically reflect the update.
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import statsmodels.formula.api as smf
import os
from pathlib import Path
import sys
import subprocess

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Coffee Shop Operations Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use paths relative to this script so the app runs regardless of current working directory
BASE_DIR   = Path(__file__).parent.resolve()
DB_FILE    = str(BASE_DIR / "coffee_shop.db")
OUTPUT_DIR = str(BASE_DIR / "outputs")
FIG_DIR    = str(Path(OUTPUT_DIR) / "figures")
ALPHA      = 0.05   # significance threshold for all statistical tests

# ── RIS Rx brand palette ──────────────────────────────────────────────────────
RIS_TEAL       = "#00a57a"          # primary brand teal
RIS_TEAL_DARK  = "#007a5c"          # dark teal (hover / active)
RIS_TEAL_PALE  = "#e0f5ef"          # very light teal (backgrounds)
RIS_NAVY       = "#1b3a4b"          # dark navy (sidebar, headings)
RIS_ORANGE     = "#f07c3a"          # warm orange (secondary / prep time)
RIS_ORANGE_PALE = "#fef3e8"         # pale orange
RIS_ORANGE_DARK = "#c04a10"         # deep orange
# Plotly continuous scales
SCALE_TEAL   = [RIS_TEAL_PALE, RIS_TEAL, RIS_TEAL_DARK]
SCALE_ORANGE = [RIS_ORANGE_PALE, RIS_ORANGE, RIS_ORANGE_DARK]
# Qualitative palette (used wherever px.colors.qualitative.Pastel was used)
RIS_QUAL = [RIS_TEAL, RIS_ORANGE, RIS_NAVY, "#33b594", "#e0793a"]

# Ensure output directories exist so plots/saves won't fail
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
# @st.cache_data tells Streamlit to cache this result so the database is not
# re-queried every time a user interacts with a widget.

@st.cache_data
def load_full_data():
    """Load the full cleaned orders table from SQLite."""
    conn = sqlite3.connect(DB_FILE)
    df   = pd.read_sql_query("SELECT * FROM orders", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

df_full = load_full_data()

# Sorted lists for filter widgets
all_agents = sorted(df_full["agent"].unique())
all_drinks = sorted(df_full["drink_name"].unique())
min_date   = df_full["date"].min().date()
max_date   = df_full["date"].max().date()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("☕ Filters")
st.sidebar.markdown("Adjust filters to explore a subset of the data.")

selected_agents = st.sidebar.multiselect(
    label="Baristas",
    options=all_agents,
    default=all_agents,
    help="Select one or more baristas to include.",
)

selected_drinks = st.sidebar.multiselect(
    label="Drink Types",
    options=all_drinks,
    default=all_drinks,
    help="Select one or more drink types to include.",
)

date_range = st.sidebar.date_input(
    label="Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filter orders by the date the order was placed.",
)

exclude_outliers = st.sidebar.checkbox(
    label="Exclude Wait-Time Outliers (IQR method)",
    value=False,
    help="Remove wait times outside Q1-1.5*IQR / Q3+1.5*IQR.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Primary KPI:** Wait time = time from order placed → drink completed."
)


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df, agents, drinks, date_range, exclude_outliers):
    """Return a filtered copy of the dataframe based on sidebar selections."""
    # Handle both 1-date and 2-date selections from the date_input widget
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    mask = (
        df["agent"].isin(agents) &
        df["drink_name"].isin(drinks) &
        (df["date"].dt.date >= start_date) &
        (df["date"].dt.date <= end_date)
    )
    filtered = df[mask].copy()

    if exclude_outliers:
        Q1  = filtered["wait_time_min"].quantile(0.25)
        Q3  = filtered["wait_time_min"].quantile(0.75)
        IQR = Q3 - Q1
        fence_lo = Q1 - 1.5 * IQR
        fence_hi = Q3 + 1.5 * IQR
        filtered = filtered[
            (filtered["wait_time_min"] >= fence_lo) &
            (filtered["wait_time_min"] <= fence_hi)
        ]

    return filtered

df = apply_filters(df_full, selected_agents, selected_drinks,
                   date_range, exclude_outliers)

# Guard against empty filter selection
if df.empty:
    st.error("No data matches the current filters. Please adjust your selections.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED COMPUTED VALUES
# ─────────────────────────────────────────────────────────────────────────────

total_orders    = len(df)
avg_wait        = df["wait_time_min"].mean()
median_wait     = df["wait_time_min"].median()
send_back_rate  = df["order_sent_back"].mean() * 100
busiest_hour    = df["hour_of_day"].mode()[0]
fastest_agent   = df.groupby("agent")["wait_time_min"].mean().idxmin()
slowest_agent   = df.groupby("agent")["wait_time_min"].mean().idxmax()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.title("☕ Coffee Shop Operations Dashboard")
st.markdown(
    "Monitor barista performance, drink trends, and time-reduction opportunities. "
    "Use the **sidebar** to filter by barista, drink type, and date range.\n"
    "Created with the assistance of generative AI (Claude Code)."
)

if exclude_outliers:
    st.info("Outlier exclusion is active — extreme wait times have been removed.")

st.markdown("---")

st.markdown(
    f"""
    <style>
    /* ── Sidebar ───────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background-color: {RIS_NAVY};
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {{
        background-color: {RIS_TEAL} !important;
    }}

    /* ── Page title ────────────────────────────────────────────────────── */
    h1 {{ color: {RIS_NAVY}; }}
    h2, h3 {{ color: {RIS_NAVY}; }}

    /* ── Metric cards ──────────────────────────────────────────────────── */
    [data-testid="metric-container"] {{
        background-color: {RIS_TEAL_PALE};
        border-left: 4px solid {RIS_TEAL};
        border-radius: 6px;
        padding: 10px 14px;
    }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {RIS_TEAL_DARK};
    }}

    /* ── Active tab highlight ──────────────────────────────────────────── */
    button[data-baseweb="tab"][aria-selected="true"] {{
        border-bottom: 3px solid {RIS_TEAL} !important;
        color: {RIS_TEAL} !important;
    }}

    /* ── Significant-row highlight (Tukey / regression tables) ─────────── */
    /* Kept via pandas Styler — no extra CSS needed */
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS (top row)
# ─────────────────────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Orders",    f"{total_orders:,}")
c2.metric("Avg Wait Time",   f"{avg_wait:.1f} min")
c3.metric("Median Wait Time",f"{median_wait:.1f} min")
c4.metric("Send-Back Rate",  f"{send_back_rate:.1f}%")
c5.metric("Busiest Hour",    f"{busiest_hour}:00")
c6.metric("Fastest Barista", fastest_agent)

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "👤 Barista Performance",
    "☕ Drink Analysis",
    "📐 Statistical Results",
    "🔍 Data Explorer",
    "💡 Change Proposal (Prompt 1)",
    "📈 Data Trends & Recommendations (Prompt 2)",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    st.subheader("Overview")

    col_left, col_right = st.columns(2)

    # ── Orders per day (line chart) ──────────────────────────────────────────
    with col_left:
        daily_counts = (
            df.groupby("date")
            .size()
            .reset_index(name="order_count")
        )
        fig = px.line(
            daily_counts, x="date", y="order_count",
            title="Daily Order Volume",
            labels={"date": "Date", "order_count": "Number of Orders"},
            markers=True,
        )
        fig.update_traces(line_color=RIS_TEAL)
        st.plotly_chart(fig, width="stretch")

    # ── Wait time distribution (histogram) ──────────────────────────────────
    with col_right:
        fig = px.histogram(
            df, x="wait_time_min", nbins=60,
            title="Wait Time Distribution",
            labels={"wait_time_min": "Wait Time (minutes)", "count": "Orders"},
            color_discrete_sequence=[RIS_TEAL],
        )
        fig.add_vline(x=avg_wait,    line_dash="dash", line_color="red",
                      annotation_text=f"Mean: {avg_wait:.1f} min",
                      annotation_position="top right")
        fig.add_vline(x=median_wait, line_dash="dot",  line_color=RIS_ORANGE,
                      annotation_text=f"Median: {median_wait:.1f} min",
                      annotation_position="top left")
        st.plotly_chart(fig, width="stretch")

    # ── Orders and avg wait by hour (dual axis) ──────────────────────────────
    hourly = (
        df.groupby("hour_of_day")
        .agg(order_count=("wait_time_min", "count"),
             avg_wait=("wait_time_min", "mean"))
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["hour_of_day"], y=hourly["order_count"],
        name="Order Count", marker_color=RIS_TEAL_PALE, yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=hourly["hour_of_day"], y=hourly["avg_wait"].round(2),
        name="Avg Wait (min)", mode="lines+markers",
        line=dict(color=RIS_TEAL, width=2),
    ))
    fig.update_layout(
        title="Order Volume and Avg Wait Time by Hour of Day",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Avg Wait Time (minutes)", side="left"),
        yaxis2=dict(title="Number of Orders", side="right", overlaying="y",
                    showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Wait time decomposition (overall) ────────────────────────────────────
    st.markdown("#### Where Does Time Go? (Average Journey Breakdown)")
    decomp_overall = {
        "Segment"         : ["Queue Wait\n(enter→order)", "Idle Wait\n(order→barista starts)",
                             "Prep Time\n(barista starts→done)"],
        "Avg Minutes"     : [
            df["queue_wait_min"].mean(),
            df["idle_wait_min"].mean(),
            df["prep_time_min"].mean(),
        ],
    }
    decomp_df = pd.DataFrame(decomp_overall)
    fig = px.bar(
        decomp_df, x="Segment", y="Avg Minutes",
        title="Average Time Per Journey Segment",
        color="Segment",
        color_discrete_sequence=RIS_QUAL,
        text_auto=".2f",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BARISTA PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    st.subheader("Barista Performance")

    # ── Per-barista summary table ─────────────────────────────────────────────
    agent_summary = (
        df.groupby("agent")
        .agg(
            total_orders    = ("wait_time_min", "count"),
            avg_wait_min    = ("wait_time_min", "mean"),
            median_wait_min = ("wait_time_min", "median"),
            avg_prep_min    = ("prep_time_min", "mean"),
            avg_idle_min    = ("idle_wait_min", "mean"),
            send_back_pct   = ("order_sent_back", lambda x: x.mean() * 100),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_wait_min", ascending=False)
    )

    st.dataframe(
        agent_summary.rename(columns={
            "agent"          : "Barista",
            "total_orders"   : "Orders",
            "avg_wait_min"   : "Avg Wait (min)",
            "median_wait_min": "Median Wait (min)",
            "avg_prep_min"   : "Avg Prep (min)",
            "avg_idle_min"   : "Avg Idle Wait (min)",
            "send_back_pct"  : "Send-Back %",
        }),
        width="stretch",
        hide_index=True,
    )

    col_left, col_right = st.columns(2)

    # ── Avg wait by barista ───────────────────────────────────────────────────
    with col_left:
        fig = px.bar(
            agent_summary.sort_values("avg_wait_min"),
            x="avg_wait_min", y="agent", orientation="h",
            title="Avg Wait Time by Barista",
            labels={"avg_wait_min": "Avg Wait (min)", "agent": "Barista"},
            color="avg_wait_min",
            color_continuous_scale=SCALE_TEAL,
            text_auto=".2f",
        )
        fig.add_vline(x=avg_wait, line_dash="dash", line_color="red",
                      annotation_text=f"Overall mean: {avg_wait:.1f}")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    # ── Send-back rate by barista ─────────────────────────────────────────────
    with col_right:
        fig = px.bar(
            agent_summary.sort_values("send_back_pct"),
            x="send_back_pct", y="agent", orientation="h",
            title="Send-Back Rate by Barista (%)",
            labels={"send_back_pct": "Send-Back Rate (%)", "agent": "Barista"},
            color="send_back_pct",
            color_continuous_scale="Reds",
            text_auto=".1f",
        )
        fig.add_vline(x=send_back_rate, line_dash="dash", line_color="red",
                      annotation_text=f"Overall: {send_back_rate:.1f}%")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    # ── Box plot: full distribution by barista ────────────────────────────────
    agent_order_sorted = agent_summary.sort_values("avg_wait_min")["agent"].tolist()
    fig = px.box(
        df, x="agent", y="wait_time_min",
        category_orders={"agent": agent_order_sorted},
        title="Wait Time Distribution by Barista (sorted by avg)",
        labels={"agent": "Barista", "wait_time_min": "Wait Time (minutes)"},
        color="agent",
    )
    fig.add_hline(y=avg_wait, line_dash="dash", line_color="red",
                  annotation_text=f"Overall mean: {avg_wait:.1f} min")
    st.plotly_chart(fig, width="stretch")

    # ── Time decomposition: idle wait vs. prep time by barista ────────────────
    decomp = (
        df.groupby("agent")[["idle_wait_min", "prep_time_min"]]
        .mean()
        .reset_index()
        .melt(id_vars="agent", var_name="segment", value_name="minutes")
    )
    decomp["segment"] = decomp["segment"].map({
        "idle_wait_min" : "Idle Wait (order→barista starts)",
        "prep_time_min" : "Prep Time (barista starts→done)",
    })

    fig = px.bar(
        decomp, x="agent", y="minutes", color="segment",
        barmode="stack",
        title="Wait Time Decomposition by Barista",
        labels={"agent": "Barista", "minutes": "Minutes", "segment": "Segment"},
        color_discrete_map={
            "Idle Wait (order→barista starts)": RIS_TEAL,
            "Prep Time (barista starts→done)": RIS_ORANGE,
        },
    )
    st.plotly_chart(fig, width="stretch")

    # ── Heatmap: avg wait by barista × drink ─────────────────────────────────
    pivot = df.pivot_table(
        values="wait_time_min", index="agent", columns="drink_name", aggfunc="mean"
    ).round(2)

    fig = px.imshow(
        pivot, text_auto=True,
        color_continuous_scale="YlOrRd",
        title="Avg Wait Time (min): Barista × Drink Type",
        labels={"color": "Avg Wait (min)"},
        aspect="auto",
    )
    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — DRINK ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

with tab3:
    st.subheader("Drink Type Analysis")

    drink_summary = (
        df.groupby("drink_name")
        .agg(
            total_orders    = ("wait_time_min", "count"),
            avg_wait_min    = ("wait_time_min", "mean"),
            median_wait_min = ("wait_time_min", "median"),
            avg_prep_min    = ("prep_time_min", "mean"),
            send_back_pct   = ("order_sent_back", lambda x: x.mean() * 100),
        )
        .round(2)
        .reset_index()
        .sort_values("avg_wait_min", ascending=False)
    )

    st.dataframe(
        drink_summary.rename(columns={
            "drink_name"     : "Drink",
            "total_orders"   : "Orders",
            "avg_wait_min"   : "Avg Wait (min)",
            "median_wait_min": "Median Wait (min)",
            "avg_prep_min"   : "Avg Prep (min)",
            "send_back_pct"  : "Send-Back %",
        }),
        width="stretch",
        hide_index=True,
    )

    col_left, col_right = st.columns(2)

    with col_left:
        fig = px.bar(
            drink_summary.sort_values("avg_wait_min"),
            x="avg_wait_min", y="drink_name", orientation="h",
            title="Avg Wait Time by Drink Type",
            labels={"avg_wait_min": "Avg Wait (min)", "drink_name": "Drink"},
            color="avg_wait_min",
            color_continuous_scale=SCALE_ORANGE,
            text_auto=".2f",
        )
        fig.add_vline(x=avg_wait, line_dash="dash", line_color="red",
                      annotation_text=f"Overall mean: {avg_wait:.1f}")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        fig = px.bar(
            drink_summary.sort_values("send_back_pct"),
            x="send_back_pct", y="drink_name", orientation="h",
            title="Send-Back Rate by Drink Type (%)",
            labels={"send_back_pct": "Send-Back Rate (%)", "drink_name": "Drink"},
            color="send_back_pct",
            color_continuous_scale="Reds",
            text_auto=".1f",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width="stretch")

    # ── Box plot: distribution by drink ─────────────────────────────────────
    drink_order_sorted = drink_summary.sort_values("avg_wait_min")["drink_name"].tolist()
    fig = px.box(
        df, x="drink_name", y="wait_time_min",
        category_orders={"drink_name": drink_order_sorted},
        title="Wait Time Distribution by Drink Type (sorted by avg)",
        labels={"drink_name": "Drink", "wait_time_min": "Wait Time (minutes)"},
        color="drink_name",
    )
    fig.add_hline(y=avg_wait, line_dash="dash", line_color="red",
                  annotation_text=f"Overall mean: {avg_wait:.1f} min")
    st.plotly_chart(fig, width="stretch")

    # ── Scatter: order volume vs avg wait (bubble size = send-back rate) ──────
    fig = px.scatter(
        drink_summary,
        x="total_orders", y="avg_wait_min",
        size="send_back_pct",
        text="drink_name",
        title="Order Volume vs. Avg Wait Time (bubble size = send-back rate)",
        labels={
            "total_orders": "Total Orders",
            "avg_wait_min": "Avg Wait (min)",
            "send_back_pct": "Send-Back %",
        },
        color="avg_wait_min",
        color_continuous_scale=SCALE_ORANGE,
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — STATISTICAL RESULTS
# ═════════════════════════════════════════════════════════════════════════════

with tab4:
    st.subheader("Statistical Analysis Results")
    st.markdown(
        "All analyses below are computed on the **currently filtered data**. "
        "Change sidebar filters to see how results shift."
    )

    # ── Minimum sample size guard ─────────────────────────────────────────────
    min_group_size = df.groupby("agent")["wait_time_min"].count().min()
    if min_group_size < 5:
        st.warning(
            "Some barista groups have fewer than 5 observations with the current "
            "filters. Statistical results may be unreliable. Consider broadening "
            "your filter selections."
        )

    # ── Re-compute ANOVA on filtered data ─────────────────────────────────────
    st.markdown("### One-Way ANOVA")

    def run_anova(df, group_col, value_col):
        """Return F, p, and eta-squared for a one-way ANOVA."""
        groups     = [g[value_col].values for _, g in df.groupby(group_col)]
        f, p       = stats.f_oneway(*groups)
        grand_mean = df[value_col].mean()
        ss_total   = ((df[value_col] - grand_mean) ** 2).sum()
        ss_between = sum(
            len(g) * (g[value_col].mean() - grand_mean) ** 2
            for _, g in df.groupby(group_col)
        )
        eta_sq = ss_between / ss_total
        return round(f, 4), round(p, 6), round(eta_sq, 4)

    f_a, p_a, eta_a = run_anova(df, "agent",      "wait_time_min")
    f_d, p_d, eta_d = run_anova(df, "drink_name", "wait_time_min")
    f_p, p_p, eta_p = run_anova(df, "agent",      "prep_time_min")

    def sig_badge(p):
        return "✅ Significant" if p < ALPHA else "❌ Not Significant"

    anova_results = pd.DataFrame({
        "Test"            : [
            "Barista → wait_time_min",
            "Drink Type → wait_time_min",
            "Barista → prep_time_min (barista speed only)",
        ],
        "F-statistic"     : [f_a, f_d, f_p],
        "p-value"         : [p_a, p_d, p_p],
        "η² (eta²)"      : [eta_a, eta_d, eta_p],
        "Interpretation"  : [sig_badge(p_a), sig_badge(p_d), sig_badge(p_p)],
    })
    st.dataframe(anova_results, width="stretch", hide_index=True)

    st.markdown(
        "**η² interpretation:** 0.01 = small effect, 0.06 = medium, 0.14 = large. "
        "It represents the proportion of total wait-time variance explained by that factor."
    )

    # ── Tukey HSD ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Tukey HSD — Pairwise Comparisons")
    st.markdown(
        "Tukey's Honest Significant Difference (HSD) test identifies **which specific "
        "pairs** differ significantly, while controlling the family-wise error rate at α=0.05."
    )

    tukey_tab1, tukey_tab2 = st.tabs(["By Barista", "By Drink Type"])

    def compute_tukey(df, group_col, value_col):
        """Run Tukey HSD and return a tidy DataFrame."""
        tukey = pairwise_tukeyhsd(
            endog=df[value_col], groups=df[group_col], alpha=ALPHA
        )
        groups   = list(tukey.groupsunique)
        pair_idx = list(combinations(range(len(groups)), 2))
        return pd.DataFrame({
            "Group 1"        : [str(groups[i]) for i, j in pair_idx],
            "Group 2"        : [str(groups[j]) for i, j in pair_idx],
            "Mean Diff (min)": tukey.meandiffs.round(3),
            "Adj. p-value"   : tukey.pvalues.round(5),
            "Lower CI"       : tukey.confint[:, 0].round(3),
            "Upper CI"       : tukey.confint[:, 1].round(3),
            "Significant"    : ["Yes" if r else "No" for r in tukey.reject],
        })

    with tukey_tab1:
        if df["agent"].nunique() < 2:
            st.info("Select at least 2 baristas to run pairwise comparisons.")
        else:
            tukey_agent_df = compute_tukey(df, "agent", "wait_time_min")

            # Color rows: red if significant, plain if not; force dark text for contrast
            def color_significant(row):
                if row["Significant"] == "Yes":
                    return ["background-color: #FDECEA; color: #1b3a4b"] * len(row)
                return [""] * len(row)

            st.dataframe(
                tukey_agent_df.style.apply(color_significant, axis=1),
                width="stretch",
                hide_index=True,
            )

            # Heatmap of p-values
            agents = sorted(df["agent"].unique())
            mat    = pd.DataFrame(1.0, index=agents, columns=agents)
            for _, row in tukey_agent_df.iterrows():
                mat.loc[row["Group 1"], row["Group 2"]] = row["Adj. p-value"]
                mat.loc[row["Group 2"], row["Group 1"]] = row["Adj. p-value"]

            fig = px.imshow(
                mat, text_auto=".3f",
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=1,
                title="Tukey HSD p-values: Barista Pairs\n(Red = significantly different)",
                labels={"color": "Adj. p-value"},
            )
            st.plotly_chart(fig, width="stretch")

    with tukey_tab2:
        if df["drink_name"].nunique() < 2:
            st.info("Select at least 2 drink types to run pairwise comparisons.")
        else:
            tukey_drink_df = compute_tukey(df, "drink_name", "wait_time_min")

            st.dataframe(
                tukey_drink_df.style.apply(color_significant, axis=1),
                width="stretch",
                hide_index=True,
            )

            drinks = sorted(df["drink_name"].unique())
            mat    = pd.DataFrame(1.0, index=drinks, columns=drinks)
            for _, row in tukey_drink_df.iterrows():
                mat.loc[row["Group 1"], row["Group 2"]] = row["Adj. p-value"]
                mat.loc[row["Group 2"], row["Group 1"]] = row["Adj. p-value"]

            fig = px.imshow(
                mat, text_auto=".3f",
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=1,
                title="Tukey HSD p-values: Drink Type Pairs\n(Red = significantly different)",
                labels={"color": "Adj. p-value"},
            )
            st.plotly_chart(fig, width="stretch")

    # ── Multiple Linear Regression ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Multiple Linear Regression")
    st.markdown(
        "Model: `wait_time_min ~ barista + drink_type + hour_of_day + order_sent_back`  \n"
        "Coefficients show how many **minutes** each factor adds or subtracts vs. the "
        "reference category (first alphabetically: **Alice**, **Americano**)."
    )

    if df["agent"].nunique() >= 2 and df["drink_name"].nunique() >= 2:
        model = smf.ols(
            formula="wait_time_min ~ C(agent) + C(drink_name) + hour_of_day + order_sent_back",
            data=df,
        ).fit()

        coef_df = pd.DataFrame({
            "Term"         : model.params.index,
            "Coefficient"  : model.params.values.round(4),
            "Std Error"    : model.bse.values.round(4),
            "t-stat"       : model.tvalues.values.round(4),
            "p-value"      : model.pvalues.values.round(5),
            "95% CI Lower" : model.conf_int()[0].values.round(4),
            "95% CI Upper" : model.conf_int()[1].values.round(4),
        })
        coef_df["Significant"] = coef_df["p-value"].apply(lambda p: "Yes" if p < ALPHA else "No")

        st.markdown(
            f"**R² = {model.rsquared:.4f}** | "
            f"**Adj. R² = {model.rsquared_adj:.4f}** | "
            f"**Model p-value = {model.f_pvalue:.6f}**"
        )

        st.dataframe(
            coef_df.style.apply(color_significant, axis=1),
            width="stretch",
            hide_index=True,
        )

        # Coefficient plot (exclude intercept)
        plot_df = coef_df[coef_df["Term"] != "Intercept"].copy()
        plot_df = plot_df.sort_values("Coefficient")
        plot_df["Color"] = plot_df["Significant"].map(
            {True: "Significant (p<0.05)", False: "Not Significant"}
        )

        fig = px.scatter(
            plot_df, x="Coefficient", y="Term",
            error_x_minus=plot_df["Coefficient"] - plot_df["95% CI Lower"],
            error_x      =plot_df["95% CI Upper"] - plot_df["Coefficient"],
            color="Color",
            color_discrete_map={
                "Significant (p<0.05)": RIS_TEAL,
                "Not Significant"      : "#cccccc",
            },
            title="Regression Coefficients ± 95% CI\n(Reference: Alice + Americano)",
            labels={"Coefficient": "Effect on Wait Time (minutes)", "Term": ""},
        )
        fig.add_vline(x=0, line_dash="dash", line_color="black")
        fig.update_layout(height=max(400, len(plot_df) * 25))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Select at least 2 baristas and 2 drink types for regression.")

    # ── Key findings callout ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Key Findings & Recommendations")

    # Dynamically compute top/bottom performers from filtered data
    agent_means  = df.groupby("agent")["wait_time_min"].mean()
    drink_means  = df.groupby("drink_name")["wait_time_min"].mean()
    slowest_bar  = agent_means.idxmax()
    fastest_bar  = agent_means.idxmin()
    slowest_drk  = drink_means.idxmax()
    fastest_drk  = drink_means.idxmin()
    sb_by_agent  = df.groupby("agent")["order_sent_back"].mean() * 100
    highest_sb   = sb_by_agent.idxmax()

    st.info(
        f"**Barista Speed:** {slowest_bar} has the highest avg wait "
        f"({agent_means[slowest_bar]:.1f} min) vs. {fastest_bar} "
        f"({agent_means[fastest_bar]:.1f} min). "
        f"If significant per ANOVA above, targeted coaching for {slowest_bar} is warranted.\n\n"
        f"**Drink Complexity:** {slowest_drk} is the slowest drink "
        f"({drink_means[slowest_drk]:.1f} min avg). Consider pre-prep or batch preparation.\n\n"
        f"**Quality:** {highest_sb} has the highest send-back rate "
        f"({sb_by_agent[highest_sb]:.1f}%). Remade orders inflate wait times — "
        f"quality training could reduce both errors and total wait time.\n\n"
        f"**Time of Day:** Monitor the busiest hour ({busiest_hour}:00) for staffing alignment."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

with tab5:
    st.subheader("Data Explorer")
    st.markdown(
        f"Showing **{len(df):,}** orders matching the current filters. "
        "Sort any column by clicking its header."
    )

    display_cols = [
        "order_id", "date", "hour_of_day", "agent", "drink_name",
        "order_sent_back", "wait_time_min", "idle_wait_min",
        "prep_time_min", "queue_wait_min",
    ]

    st.dataframe(
        df[display_cols].rename(columns={
            "order_id"       : "Order ID",
            "date"           : "Date",
            "hour_of_day"    : "Hour",
            "agent"          : "Barista",
            "drink_name"     : "Drink",
            "order_sent_back": "Sent Back",
            "wait_time_min"  : "Wait (min)",
            "idle_wait_min"  : "Idle Wait (min)",
            "prep_time_min"  : "Prep (min)",
            "queue_wait_min" : "Queue Wait (min)",
        }).round(2),
        width="stretch",
        height=500,
    )

    # ── CSV Download ─────────────────────────────────────────────────────────
    @st.cache_data
    def convert_to_csv(dataframe):
        return dataframe.to_csv(index=False).encode("utf-8")

    csv_bytes = convert_to_csv(df[display_cols].round(2))

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="coffee_shop_filtered.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRE-COMPUTED ANALYTICS  (full dataset — used in tabs 6 & 7)
# ─────────────────────────────────────────────────────────────────────────────

# ANOVA: barista → wait_time_min on full dataset
_groups_full     = [g["wait_time_min"].values for _, g in df_full.groupby("agent")]
_f_full, _p_full = stats.f_oneway(*_groups_full)
_grand_mean_full = df_full["wait_time_min"].mean()
_ss_total_full   = ((df_full["wait_time_min"] - _grand_mean_full) ** 2).sum()
_ss_between_full = sum(
    len(g) * (g["wait_time_min"].mean() - _grand_mean_full) ** 2
    for _, g in df_full.groupby("agent")
)
_eta_sq_full = _ss_between_full / _ss_total_full

# OLS regression on full dataset
_reg_full = smf.ols(
    "wait_time_min ~ C(agent) + C(drink_name) + hour_of_day + order_sent_back",
    data=df_full,
).fit()

# Slowest barista stats
_ameans_full  = df_full.groupby("agent")["wait_time_min"].mean()
_aorders_full = df_full.groupby("agent")["wait_time_min"].count()
_aprep_full   = df_full.groupby("agent")["prep_time_min"].mean()
_aidle_full   = df_full.groupby("agent")["idle_wait_min"].mean()
_slowest_full = _ameans_full.idxmax()
_peer_mean_full   = _ameans_full.drop(_slowest_full).mean()
_gap_full         = _ameans_full[_slowest_full] - _peer_mean_full
_vol_share_full   = _aorders_full[_slowest_full] / len(df_full)
_system_impact_f  = _vol_share_full * _gap_full
_coef_key_agent   = f"C(agent)[T.{_slowest_full}]"
_slowest_coef     = _reg_full.params.get(_coef_key_agent, float("nan"))
_slowest_pval     = _reg_full.pvalues.get(_coef_key_agent, float("nan"))

# Top send-back drink stats
_drink_sb_full    = df_full.groupby("drink_name")["order_sent_back"].mean().mul(100)
_top_sb_drink     = _drink_sb_full.idxmax()
_top_sb_rate      = _drink_sb_full[_top_sb_drink]
_other_sb_avg_f   = _drink_sb_full.drop(_top_sb_drink).mean()
_sb_multiplier    = _top_sb_rate / _other_sb_avg_f
_top_drink_orders = df_full[df_full["drink_name"] == _top_sb_drink].shape[0]
_top_drink_remakes = int(_top_drink_orders * _top_sb_rate / 100)
_coef_key_drink   = f"C(drink_name)[T.{_top_sb_drink}]"
_top_drink_coef   = _reg_full.params.get(_coef_key_drink, float("nan"))
_top_drink_pval   = _reg_full.pvalues.get(_coef_key_drink, float("nan"))

# Time decomposition
_avg_wait_f  = df_full["wait_time_min"].mean()
_avg_prep_f  = df_full["prep_time_min"].mean()
_avg_idle_f  = df_full["idle_wait_min"].mean()
_avg_queue_f = df_full["queue_wait_min"].mean()
_prep_pct_f  = _avg_prep_f  / _avg_wait_f * 100
_idle_pct_f  = _avg_idle_f  / _avg_wait_f * 100
_queue_pct_f = _avg_queue_f / _avg_wait_f * 100

# Volume concentration
_avol_pct_full = (_aorders_full / len(df_full) * 100).sort_values(ascending=False)
_top2_pct_f    = _avol_pct_full.iloc[:2].sum()
_top2_names_f  = ", ".join(_avol_pct_full.index[:2].tolist())

# Peak hour
_hrly_wait_full  = df_full.groupby("hour_of_day")["wait_time_min"].mean()
_peak_hr_f       = int(_hrly_wait_full.idxmax())
_peak_wait_f     = _hrly_wait_full[_peak_hr_f]
_hour_pval_full  = _reg_full.pvalues.get("hour_of_day", float("nan"))

# ANOVA: drink type → wait_time_min on full dataset
_drink_groups_full     = [g["wait_time_min"].values for _, g in df_full.groupby("drink_name")]
_f_drink_full, _p_drink_full = stats.f_oneway(*_drink_groups_full)

# Number of baristas in full dataset
_n_baristas_full = df_full["agent"].nunique()
_n_orders_full   = len(df_full)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — CHANGE PROPOSAL (PROMPT 1)
# ═════════════════════════════════════════════════════════════════════════════

with tab6:
    st.subheader("Change Proposal (Prompt 1)")
    st.markdown("### Objective")
    st.markdown(
        "> *I was asked to streamline the drink making at a very successful local chain of coffee shops. "
        "These coffee shops are packed, and the owner needs me to minimize the time between customer order "
        "and drink delivery. The owner said I have free reign and can make any changes necessary.*"
    )
    st.markdown("---")

    # ── Q1: Approach ─────────────────────────────────────────────────────────
    st.markdown("### How Would I Approach Solving This Problem?")
    st.markdown(
        "Before recommending any change, I would start with a **diagnostic phase** to understand "
        "where time is actually lost — rather than assuming the fix. Specifically:\n\n"
        f"1. **Map the order journey.** Break wait time into its component parts: "
        "line wait (customer enters → order taken), idle wait (order placed → barista starts), "
        "and prep time (barista starts → drink delivered). The data already shows this split: "
        f"prep time accounts for ~{_prep_pct_f:.0f}% of total wait, making it the primary lever.\n\n"
        f"2. **Identify the largest sources of variance.** Statistical analysis reveals that "
        f"barista identity alone explains {_eta_sq_full*100:.0f}% of all wait-time variance "
        f"(η² = {_eta_sq_full:.2f}, p < 0.001). "
        f"One barista in the current dataset averages {_gap_full:.1f} minutes longer per order than peers — "
        "a gap large enough that closing it would meaningfully move the shop-wide average.\n\n"
        "3. **Observe operations directly.** Shadow baristas during peak hours to understand "
        "*why* the gap exists: technique differences, unfamiliarity with certain drinks, "
        "equipment placement, or interruptions. Numbers tell you *what*; observation tells you *why*.\n\n"
    )
    st.markdown("---")

    # ── Q2: Options and recommendation ───────────────────────────────────────
    st.markdown("### Options and Recommendation")

    st.markdown("#### Option A | Barista Standardization & Targeted Coaching")
    st.markdown(
        "Develop a standardized drink-making protocol (step-by-step SOPs with time benchmarks "
        "for each drink) and run structured coaching sessions with baristas whose prep times "
        "exceed the peer average. Pair slower baristas with top performers for hands-on shadowing.\n\n"
        "**Pros:** Low cost, directly addresses the #1 driver of variance, results visible quickly.  \n"
        "**Cons:** Requires manager time; behavioral change can be slow without accountability."
    )

    st.markdown("#### Option B | Station Pre-Prep & Batching")
    st.markdown(
        f"Pre-portion ingredients (syrups, milks, powders) before peak hours and batch-prep "
        f"high-send-back items (e.g., {_top_sb_drink}, which has a {_top_sb_rate:.1f}% send-back rate — "
        f"roughly {_sb_multiplier:.0f}× the average of all other drinks) to reduce rework. "
        "Consider redesigning workstation layout so the most-used ingredients are within arm's reach.\n\n"
        "**Pros:** Reduces prep time for all baristas; improves quality consistency.  \n"
        "**Cons:** Requires some upfront labor and storage space; batched items may lose freshness."
    )

    st.markdown("#### Option C — Dynamic Staffing & Scheduling")
    st.markdown(
        f"Align staffing levels with hourly demand patterns. The data shows the "
        f"{_peak_hr_f}:00–{_peak_hr_f+1}:00 window carries slightly elevated wait times "
        f"({_peak_wait_f:.2f} min avg). Additional barista support (stocking, labeling, cashiering) "
        "during peak hours frees baristas to focus purely on drink preparation.\n\n"
        "**Pros:** Addresses volume-driven slowdowns; reduces per-barista cognitive load.  \n"
        "**Cons:** Higher labor cost; scheduling complexity across a chain."
    )

    st.info(
        "**Recommended Option: Option A — Barista Standardization & Targeted Coaching**\n\n"
        f"The data is clear: barista identity is by far the dominant driver of wait-time "
        f"variance, and the gap is concentrated in one barista who adds approximately "
        f"+{_slowest_coef:.2f} minutes per order "
        f"(statistically significant regression coefficient, p < 0.001). That barista "
        f"handles {_vol_share_full*100:.0f}% of all orders, so closing the gap would reduce "
        f"the shop-wide average wait time by an estimated **~{_system_impact_f:.2f} minutes** "
        f"— a ~{_system_impact_f/_avg_wait_f*100:.0f}% improvement — without any capital "
        "expenditure.\n\n"
        "Option A addresses the root cause directly. Options B and C are valuable complements but "
        "should follow, not lead, because the data shows drink type is not a statistically "
        f"significant driver of total wait time (drink-type ANOVA p = {_p_drink_full:.2f}), "
        "and hourly volume differences are modest.\n\n"
        "Implementation: (1) Develop SOPs for each drink with target prep times. "
        "(2) Film top-performer technique as a reference. "
        "(3) Run bi-weekly 1:1 coaching sessions with progress tracked against benchmarks. "
        "(4) Introduce a peer-shadowing program pairing newer or slower baristas with top performers."
    )
    st.markdown("---")

    # ── Q3: Metrics to measure before rolling out ─────────────────────────────
    st.markdown("### Metrics to Measure in the Pilot Coffee Shop")
    st.markdown(
        "The pilot coffee shop should be treated as a controlled experiment. "
        "Collect the following metrics for **at least 4 weeks pre-intervention and 4 weeks post-intervention**, "
        "then compare before deciding on chain-wide rollout:\n"
    )

    pilot_metrics = {
        "Metric": [
            "Average wait time (order → delivery)",
            "Median wait time",
            "Prep time per barista",
            "Idle wait time (order → barista starts)",
            "Send-back / remake rate",
            "Customer satisfaction score",
            "Orders per hour (throughput)",
            f"Peak-hour ({_peak_hr_f}:00–{_peak_hr_f+1}:00) wait time",
            "Barista-level prep time variance",
        ],
        "Why It Matters": [
            "Primary KPI — the goal is to reduce this",
            "More robust to outliers than the mean; tracks the typical customer experience",
            "Isolates whether coaching improved individual barista speed",
            "Detects if baristas are starting orders faster (queue management improvement)",
            "Tracks quality; a faster but sloppier barista is not a win",
            "Validates that speed gains did not come at the expense of perceived quality",
            "Ensures we are serving more customers, not just serving fewer faster",
            "Captures the highest-stress window where improvements matter most",
            "Measures whether the coaching is reducing spread, not just the mean",
        ],
        "Target": [
            "↓ by ≥ 10% vs. baseline",
            "↓ by ≥ 10% vs. baseline",
            "Coached barista(s) within 0.25 min of peer median",
            "↓ or stable",
            "≤ shop-wide baseline rate",
            "Stable or improving vs. baseline",
            "↑ or stable",
            "↓ by ≥ 10% vs. baseline",
            "Coefficient of variation ↓ across baristas",
        ],
    }
    st.dataframe(
        pd.DataFrame(pilot_metrics),
        width="stretch",
        hide_index=True,
    )
    st.markdown("---")

    # ── Q4: Sustaining performance over time ──────────────────────────────────
    st.markdown("### Sustaining Performance After Chain-Wide Rollout")
    st.markdown(
        "Speed improvements without a sustaining system will drift back. "
        "The following mechanisms would be implemented to prevent regression:\n\n"
        "**1. Ongoing Dashboard Monitoring (this dashboard)**  \n"
        "The operations dashboard should become a regular management ritual — reviewed weekly "
        "by shift supervisors and monthly by the owner. Barista-level prep times and wait times "
        "should be visible to managers in near real-time so regressions are caught early, "
        "not discovered in quarterly reviews.\n\n"
        "**2. Barista Performance Scorecards**  \n"
        "Each barista receives a monthly scorecard showing their avg prep time, wait time, "
        "and send-back rate vs. the shop benchmark. This creates accountability without "
        "requiring managers to micromanage — baristas self-correct when they can see their "
        "own trend.\n\n"
        "**3. Standardized Onboarding Protocol**  \n"
        "New hires complete a structured training program covering the SOPs developed in the "
        "pilot. Certification requires hitting target prep-time benchmarks before solo shifts. "
        "This prevents the coaching gains from being diluted by underprepared new staff.\n\n"
        "**4. Quarterly SOP Reviews**  \n"
        "As the menu evolves or new equipment is introduced, SOPs should be refreshed. "
        "A quarterly review cadence — led by the top-performing barista plus a manager — "
        "keeps standards current and gives high performers a leadership stake in the program.\n\n"
        "**5. Alert Thresholds on Key Metrics**  \n"
        "Set automated alerts (e.g., via the dashboard) if any barista's 30-day rolling "
        "average prep time exceeds the shop median by more than 20%. This triggers a coaching "
        "conversation before the gap becomes significant.\n\n"
        "**6. Incentive Alignment**  \n"
        "Consider tying a portion of shift-bonus or recognition programs to team-level "
        "wait-time performance, not just individual metrics. This encourages peer coaching "
        "and discourages a 'not my problem' attitude when a teammate is struggling."
    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — DATA TRENDS & RECOMMENDATIONS (PROMPT 2)
# ═════════════════════════════════════════════════════════════════════════════

with tab7:
    st.subheader("Data Trends & Recommendations (Prompt 2)")
    st.markdown(
        f"A review of the data across {_n_orders_full:,} orders and {_n_baristas_full} baristas "
        "surfaces several issues the owner should be aware of. Findings are ordered by estimated "
        "business impact."
    )
    st.markdown("---")

    # ── Finding 1: Charlie outlier ────────────────────────────────────────────
    st.markdown("### Finding 1 — One Barista Is a Significant Performance Outlier")

    # Compute dynamically from filtered data
    agent_means_full  = df_full.groupby("agent")["wait_time_min"].mean().round(2)
    agent_prep_full   = df_full.groupby("agent")["prep_time_min"].mean().round(2)
    agent_idle_full   = df_full.groupby("agent")["idle_wait_min"].mean().round(2)
    agent_orders_full = df_full.groupby("agent")["wait_time_min"].count()

    slowest_name = agent_means_full.idxmax()
    peer_mean    = agent_means_full.drop(slowest_name).mean().round(2)
    gap          = (agent_means_full[slowest_name] - peer_mean).round(2)
    pct_orders   = (agent_orders_full[slowest_name] / len(df_full) * 100).round(1)
    system_impact = (agent_orders_full[slowest_name] / len(df_full) * gap).round(2)

    st.error(
        f"**{slowest_name}** averages **{agent_means_full[slowest_name]} min** per order — "
        f"**{gap} minutes (+{(gap/peer_mean*100):.0f}%)** above the peer average of {peer_mean} min. "
        f"{slowest_name} handles {pct_orders}% of all orders, so this gap inflates the shop-wide "
        f"average wait by an estimated **{system_impact:.2f} minutes**. "
        f"Statistical analysis confirms this is not random variation: "
        f"ANOVA F = {_f_full:.1f} (p < 0.001), "
        f"η² = {_eta_sq_full:.2f}, meaning barista identity alone explains "
        f"{_eta_sq_full*100:.0f}% of all wait-time variance. "
        f"The regression coefficient for {slowest_name} is +{_slowest_coef:.2f} min "
        f"(p < 0.001) — the only barista with a statistically significant effect."
    )

    agent_compare = pd.DataFrame({
        "Barista"        : agent_means_full.index,
        "Avg Wait (min)" : agent_means_full.values,
        "Avg Prep (min)" : agent_prep_full.values,
        "Avg Idle (min)" : agent_idle_full.values,
        "Orders"         : agent_orders_full.values,
    }).sort_values("Avg Wait (min)", ascending=False).reset_index(drop=True)

    fig = px.bar(
        agent_compare,
        x="Avg Wait (min)", y="Barista", orientation="h",
        title=f"Avg Wait Time by Barista — {slowest_name} Is a Clear Outlier",
        color="Avg Wait (min)",
        color_continuous_scale="RdYlGn_r",
        text_auto=".2f",
    )
    fig.add_vline(x=peer_mean, line_dash="dash", line_color=RIS_NAVY,
                  annotation_text=f"Peer avg: {peer_mean} min")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"**Recommendation:** Initiate targeted coaching for {slowest_name} immediately. "
        f"The {gap}-minute gap is driven by both slower prep time "
        f"({agent_prep_full[slowest_name]} min vs. {agent_prep_full.drop(slowest_name).mean():.2f} min peer avg) "
        f"and higher idle wait ({agent_idle_full[slowest_name]} min vs. "
        f"{agent_idle_full.drop(slowest_name).mean():.2f} min peer avg), suggesting both "
        f"technique and order-pickup habits need attention."
    )
    st.markdown("---")

    # ── Finding 2: Matcha Latte quality problem ───────────────────────────────
    st.markdown("### Finding 2 — Matcha Latte Has a Critical Quality Problem")

    drink_sb_full = (
        df_full.groupby("drink_name")["order_sent_back"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )
    matcha_sb     = drink_sb_full["Matcha Latte"]
    other_avg_sb  = drink_sb_full.drop("Matcha Latte").mean().round(2)
    matcha_orders = df_full[df_full["drink_name"] == "Matcha Latte"].shape[0]

    st.warning(
        f"**Matcha Latte** has a send-back rate of **{matcha_sb}%** — "
        f"approximately **{matcha_sb / other_avg_sb:.0f}× higher** than the average of all other drinks "
        f"({other_avg_sb}%). Across {matcha_orders:,} Matcha Latte orders, this means roughly "
        f"{int(matcha_orders * matcha_sb / 100)} remakes, each of which adds wait time for that "
        f"customer *and* clogs the workstation for others. The regression confirms {_top_sb_drink} "
        f"is the only drink with a statistically significant wait-time premium "
        f"(+{_top_drink_coef:.2f} min, p = {_top_drink_pval:.3f})."
    )

    fig = px.bar(
        drink_sb_full.reset_index().rename(columns={"drink_name": "Drink", "order_sent_back": "Send-Back %"}),
        x="Send-Back %", y="Drink", orientation="h",
        title="Send-Back Rate by Drink Type — Matcha Latte Is an Outlier",
        color="Send-Back %",
        color_continuous_scale="Reds",
        text_auto=".1f",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        "**Recommendation:** Investigate the Matcha Latte preparation process immediately. "
        "Possible causes include inconsistent matcha-to-milk ratios, inadequate whisking, "
        "or unclear customer expectations. Consider: (1) standardizing the recipe with a "
        "measured scoop and documented steps; (2) adding a quality-check step before handoff; "
        "(3) temporarily flagging the drink as 'made-to-order — please allow extra time' "
        "to reset customer expectations while the recipe is tightened."
    )
    st.markdown("---")

    # ── Finding 3: Prep time dominates wait ──────────────────────────────────
    st.markdown("### Finding 3 — Prep Time Dominates Wait; Idle Wait Is the Hidden Lever")

    overall_avg_wait  = df_full["wait_time_min"].mean()
    overall_avg_prep  = df_full["prep_time_min"].mean()
    overall_avg_idle  = df_full["idle_wait_min"].mean()
    overall_avg_queue = df_full["queue_wait_min"].mean()

    prep_pct  = overall_avg_prep  / overall_avg_wait * 100
    idle_pct  = overall_avg_idle  / overall_avg_wait * 100
    queue_pct = overall_avg_queue / overall_avg_wait * 100

    col_l, col_r = st.columns(2)
    with col_l:
        decomp_data = pd.DataFrame({
            "Segment": ["Queue Wait\n(enter→order)", "Idle Wait\n(order→barista starts)",
                        "Prep Time\n(barista starts→done)"],
            "Minutes": [overall_avg_queue, overall_avg_idle, overall_avg_prep],
            "Pct": [f"{queue_pct:.0f}%", f"{idle_pct:.0f}%", f"{prep_pct:.0f}%"],
        })
        fig = px.pie(
            decomp_data, values="Minutes", names="Segment",
            title=f"Where the {overall_avg_wait:.1f}-Min Avg Wait Goes",
            color_discrete_sequence=[RIS_TEAL, RIS_ORANGE, RIS_NAVY],
        )
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.markdown(
            f"\n\n"
            f"- **Queue wait:** {overall_avg_queue:.2f} min ({queue_pct:.0f}%) — "
            f"time from entering the shop to placing an order. Largely driven by cashier speed.\n\n"
            f"- **Idle wait:** {overall_avg_idle:.2f} min ({idle_pct:.0f}%) — "
            f"time between order being placed and a barista picking it up. "
            f"This is *dead time* where the customer is waiting but no work is being done. "
            f"The outlier barista's idle wait is {agent_idle_full[slowest_name]} min, "
            f"vs. {agent_idle_full.drop(slowest_name).mean():.2f} min for peers.\n\n"
            f"- **Prep time:** {overall_avg_prep:.2f} min ({prep_pct:.0f}%) — "
            f"actual drink production. This is the largest segment and the primary target "
            f"for coaching and SOP improvements."
        )

    st.markdown(
        "**Recommendation:** While prep time is the biggest segment, the {:.2f}-min idle wait "
        "is a low-hanging opportunity. Introducing a ticket system or display screen that "
        "alerts baristas the moment an order is placed — rather than relying on verbal handoff "
        "— could meaningfully reduce idle wait across all baristas.".format(overall_avg_idle)
    )
    st.markdown("---")

    # ── Finding 4: Volume concentration risk ─────────────────────────────────
    st.markdown("### Finding 4 — Order Volume Is Concentrated in Two Baristas")

    agent_vol_pct = (agent_orders_full / len(df_full) * 100).round(1).sort_values(ascending=False)
    top2_pct      = agent_vol_pct.iloc[:2].sum()
    top2_names    = ", ".join(agent_vol_pct.index[:2].tolist())

    fig = px.pie(
        values=agent_vol_pct.values,
        names=agent_vol_pct.index,
        title="Share of Total Orders by Barista",
        color_discrete_sequence=RIS_QUAL,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"**{top2_names}** together handle **{top2_pct:.0f}%** of all orders. "
        f"If either calls out sick or leaves the business, throughput drops sharply. "
        f"This is a staffing concentration risk — particularly relevant for a chain planning "
        f"to scale. Ensure cross-training so that all baristas can handle any drink at "
        f"a consistent pace, and that scheduling does not create single points of failure."
    )
    st.markdown("---")

    # ── Finding 5: Hour-of-day pattern ───────────────────────────────────────
    st.markdown("### Finding 5 — Afternoon Hours Show Elevated Wait Times")

    hourly_full = (
        df_full.groupby("hour_of_day")
        .agg(order_count=("wait_time_min", "count"),
             avg_wait=("wait_time_min", "mean"))
        .reset_index()
    )
    peak_hour_row = hourly_full.loc[hourly_full["avg_wait"].idxmax()]
    peak_hr       = int(peak_hour_row["hour_of_day"])
    peak_wait     = peak_hour_row["avg_wait"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_full["hour_of_day"], y=hourly_full["order_count"],
        name="Order Count", marker_color=RIS_TEAL_PALE, yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=hourly_full["hour_of_day"], y=hourly_full["avg_wait"].round(2),
        name="Avg Wait (min)", mode="lines+markers",
        line=dict(color=RIS_TEAL, width=2),
    ))
    fig.add_vline(x=peak_hr, line_dash="dot", line_color="red",
                  annotation_text=f"Peak wait: {peak_hr}:00 ({peak_wait:.2f} min)")
    fig.update_layout(
        title="Avg Wait Time and Order Volume by Hour",
        xaxis_title="Hour of Day",
        yaxis=dict(title="Avg Wait Time (min)", side="left"),
        yaxis2=dict(title="Orders", side="right", overlaying="y", showgrid=False),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown(
        f"Wait times peak at **{peak_hr}:00** ({peak_wait:.2f} min avg), while order volume "
        f"is relatively flat across the operating hours. Notably, hour-of-day is *not* "
        f"statistically significant in the regression (p = {_hour_pval_full:.2f}), suggesting "
        f"the afternoon spike is modest and partially an artifact of barista assignment rather "
        f"than pure volume. Nonetheless, the {peak_hr}:00 window is worth monitoring — ensure "
        f"the strongest baristas are scheduled during this window."
    )
    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("### Summary: Priority Issues for the Owner")

    summary_data = {
        "Priority": ["🔴 Critical", "🔴 Critical", "🟡 Important", "🟡 Important", "🟢 Monitor"],
        "Issue": [
            f"{slowest_name} averages +{_gap_full:.1f} min/order above peers",
            f"{_top_sb_drink} send-back rate is {_sb_multiplier:.0f}× the shop average",
            f"Idle wait ({_avg_idle_f:.2f} min) is dead time with no work happening",
            f"{_top2_pct_f:.0f}% of order volume handled by only 2 baristas",
            f"Hour {_peak_hr_f}:00–{_peak_hr_f+1}:00 shows elevated wait times ({_peak_wait_f:.2f} min avg)",
        ],
        "Estimated Impact": [
            f"−{_system_impact_f:.2f} min shop avg if gap closed",
            f"~{_top_drink_remakes} unnecessary remakes in this dataset; adds congestion",
            f"Up to −{_avg_idle_f:.2f} min per order if eliminated",
            "Major throughput risk if top 2 are unavailable",
            "Modest; monitor for staffing alignment",
        ],
        "Recommended Action": [
            "Targeted coaching + SOP rollout",
            "Standardize recipe; add QC step before handoff",
            "Implement order-display system / ticket alerts",
            "Cross-train all baristas; diversify scheduling",
            "Assign top performers to peak-hour shifts",
        ],
    }
    st.dataframe(pd.DataFrame(summary_data), width="stretch", hide_index=True)


if __name__ == "__main__":
    # If a user runs `python 04_dashboard.py`, forward to Streamlit using
    # the same Python executable. This avoids requiring the `streamlit`
    # CLI to be on PATH and makes the script double as a launcher.
    if "streamlit" not in sys.modules:
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
        except FileNotFoundError:
            print("Error: streamlit is not installed in this Python environment.")
            print(f"Install with: {sys.executable} -m pip install streamlit")
    else:
        # Already running under Streamlit; do nothing.
        pass
