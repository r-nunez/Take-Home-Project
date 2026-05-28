# =============================================================================
# 02_descriptive_stats.py
# PURPOSE: Compute descriptive statistics and identify outliers.
#          Produces summary CSVs and publication-ready charts.
#
# DEPENDS ON: coffee_shop.db  (created by 01_clean_data.py)
# OUTPUTS:
#   outputs/descriptive_stats_overall.csv
#   outputs/descriptive_stats_by_agent.csv
#   outputs/descriptive_stats_by_drink.csv
#   outputs/send_back_by_agent.csv
#   outputs/send_back_by_drink.csv
#   outputs/wait_time_by_hour.csv
#   outputs/outliers.csv
#   outputs/figures/  (several .png charts)
# =============================================================================

import sys
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

sys.stdout.reconfigure(encoding="utf-8")

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Seaborn global style ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

DB_FILE = "coffee_shop.db"


# =============================================================================
# STEP 1 — LOAD DATA FROM SQLITE
# =============================================================================
# We use a SQL SELECT to pull only the columns needed for descriptive analysis.
# This keeps the query readable and demonstrates SQL-based data retrieval.

print("=" * 60)
print("STEP 1: Loading data from SQLite")
print("=" * 60)

load_sql = """
    SELECT
        order_id,
        drink_name,
        agent,
        order_sent_back,
        hour_of_day,
        day_of_week,
        date,
        wait_time_min,
        queue_wait_min,
        idle_wait_min,
        prep_time_min,
        dwell_time_min
    FROM orders
"""

conn = sqlite3.connect(DB_FILE)
df   = pd.read_sql_query(load_sql, conn)
conn.close()

print(f"  Rows loaded: {len(df):,}")


# =============================================================================
# STEP 2 — OVERALL DESCRIPTIVE STATISTICS
# =============================================================================
# We examine all five time segments so we can understand WHERE in the customer
# journey time is lost, not just the total.

print("\n" + "=" * 60)
print("STEP 2: Overall descriptive statistics")
print("=" * 60)

time_metrics = [
    "wait_time_min",    # PRIMARY KPI: order placed → drink ready
    "queue_wait_min",   # enter shop → order placed
    "idle_wait_min",    # order placed → barista starts
    "prep_time_min",    # barista starts → drink ready
    "dwell_time_min",   # drink ready → customer leaves
]

# percentiles=[...] adds those quantiles to the describe() output
overall_stats = df[time_metrics].describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.95]).T

# Coefficient of variation (std / mean) measures relative variability;
# higher CV = more inconsistency in that metric.
overall_stats["cv"] = overall_stats["std"] / overall_stats["mean"]

print(overall_stats.round(2).to_string())
overall_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats_overall.csv"))


# =============================================================================
# STEP 3 — DESCRIPTIVE STATISTICS BY AGENT (SQL + pandas)
# =============================================================================
# SQL handles the grouping and basic aggregation; pandas adds percentiles
# that require custom lambdas.

print("\n" + "=" * 60)
print("STEP 3: Wait time by barista")
print("=" * 60)

# ── SQL portion: counts and simple aggregates ─────────────────────────────────
agent_sql = """
    SELECT
        agent,
        COUNT(*)                                           AS total_orders,
        ROUND(AVG(wait_time_min),  2)                      AS avg_wait_min,
        ROUND(MIN(wait_time_min),  2)                      AS min_wait_min,
        ROUND(MAX(wait_time_min),  2)                      AS max_wait_min,
        ROUND(AVG(prep_time_min),  2)                      AS avg_prep_min,
        ROUND(AVG(idle_wait_min),  2)                      AS avg_idle_min,
        ROUND(SUM(order_sent_back) * 100.0 / COUNT(*), 2) AS send_back_pct
    FROM orders
    GROUP BY agent
    ORDER BY avg_wait_min DESC;
"""
conn = sqlite3.connect(DB_FILE)
agent_sql_df = pd.read_sql_query(agent_sql, conn)
conn.close()

# ── Pandas portion: percentiles (not natively supported in SQLite) ────────────
agent_percentiles = (
    df.groupby("agent")["wait_time_min"]
    .agg(
        p25=lambda x: x.quantile(0.25),
        median=lambda x: x.quantile(0.50),
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
        std="std",
    )
    .round(2)
    .reset_index()
)

# Merge SQL results with percentile results
agent_stats = agent_sql_df.merge(agent_percentiles, on="agent")
agent_stats  = agent_stats.sort_values("avg_wait_min", ascending=False)

print(agent_stats.to_string(index=False))
agent_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats_by_agent.csv"), index=False)


# =============================================================================
# STEP 4 — DESCRIPTIVE STATISTICS BY DRINK TYPE (SQL + pandas)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Wait time by drink type")
print("=" * 60)

drink_sql = """
    SELECT
        drink_name,
        COUNT(*)                                           AS total_orders,
        ROUND(AVG(wait_time_min),  2)                      AS avg_wait_min,
        ROUND(MIN(wait_time_min),  2)                      AS min_wait_min,
        ROUND(MAX(wait_time_min),  2)                      AS max_wait_min,
        ROUND(AVG(prep_time_min),  2)                      AS avg_prep_min,
        ROUND(AVG(idle_wait_min),  2)                      AS avg_idle_min,
        ROUND(SUM(order_sent_back) * 100.0 / COUNT(*), 2) AS send_back_pct
    FROM orders
    GROUP BY drink_name
    ORDER BY avg_wait_min DESC;
"""
conn = sqlite3.connect(DB_FILE)
drink_sql_df = pd.read_sql_query(drink_sql, conn)
conn.close()

drink_percentiles = (
    df.groupby("drink_name")["wait_time_min"]
    .agg(
        p25=lambda x: x.quantile(0.25),
        median=lambda x: x.quantile(0.50),
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90),
        std="std",
    )
    .round(2)
    .reset_index()
)

drink_stats = drink_sql_df.merge(drink_percentiles, on="drink_name")
drink_stats  = drink_stats.sort_values("avg_wait_min", ascending=False)

print(drink_stats.to_string(index=False))
drink_stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats_by_drink.csv"), index=False)


# =============================================================================
# STEP 5 — SEND-BACK ANALYSIS
# =============================================================================
# order_sent_back = 1 means the drink was returned and likely remade.
# We look at (a) overall rate, (b) rate by agent, and (c) rate by drink.

print("\n" + "=" * 60)
print("STEP 5: Send-back analysis")
print("=" * 60)

overall_rate = df["order_sent_back"].mean() * 100
print(f"  Overall send-back rate: {overall_rate:.2f}%")

# Wait time comparison: sent-back vs. not sent back
comparison = df.groupby("order_sent_back")["wait_time_min"].describe().round(2)
comparison.index = ["Not sent back (0)", "Sent back (1)"]
print("\n  Wait time comparison (sent back vs. not):")
print(comparison.to_string())

# Already captured in agent_stats and drink_stats via send_back_pct column;
# save standalone CSVs for quick reference.
agent_sendback = agent_stats[["agent", "total_orders", "send_back_pct"]].sort_values(
    "send_back_pct", ascending=False
)
drink_sendback = drink_stats[["drink_name", "total_orders", "send_back_pct"]].sort_values(
    "send_back_pct", ascending=False
)
agent_sendback.to_csv(os.path.join(OUTPUT_DIR, "send_back_by_agent.csv"), index=False)
drink_sendback.to_csv(os.path.join(OUTPUT_DIR, "send_back_by_drink.csv"), index=False)

print("\n  Send-back rate by barista:")
print(agent_sendback.to_string(index=False))

print("\n  Send-back rate by drink:")
print(drink_sendback.to_string(index=False))


# =============================================================================
# STEP 6 — TIME-OF-DAY ANALYSIS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Wait time by hour of day")
print("=" * 60)

hourly_sql = """
    SELECT
        hour_of_day,
        COUNT(*)                     AS order_count,
        ROUND(AVG(wait_time_min), 2) AS avg_wait_min,
        ROUND(MIN(wait_time_min), 2) AS min_wait_min,
        ROUND(MAX(wait_time_min), 2) AS max_wait_min
    FROM orders
    GROUP BY hour_of_day
    ORDER BY hour_of_day;
"""
conn = sqlite3.connect(DB_FILE)
hourly_stats = pd.read_sql_query(hourly_sql, conn)
conn.close()

# Add median (not available in SQLite without extension)
hourly_median = df.groupby("hour_of_day")["wait_time_min"].median().reset_index()
hourly_median.columns = ["hour_of_day", "median_wait_min"]
hourly_stats = hourly_stats.merge(hourly_median, on="hour_of_day")

print(hourly_stats.to_string(index=False))
hourly_stats.to_csv(os.path.join(OUTPUT_DIR, "wait_time_by_hour.csv"), index=False)


# =============================================================================
# STEP 7 — OUTLIER ANALYSIS (IQR method)
# =============================================================================
# The IQR (Interquartile Range) method defines outliers as observations that
# fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
# This is a robust approach that is not sensitive to extreme values.

print("\n" + "=" * 60)
print("STEP 7: Outlier analysis (IQR method)")
print("=" * 60)

Q1  = df["wait_time_min"].quantile(0.25)
Q3  = df["wait_time_min"].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"  Q1          : {Q1:.2f} min")
print(f"  Q3          : {Q3:.2f} min")
print(f"  IQR         : {IQR:.2f} min")
print(f"  Lower fence : {lower_fence:.2f} min  (values below = outlier)")
print(f"  Upper fence : {upper_fence:.2f} min  (values above = outlier)")

df_outliers = df[
    (df["wait_time_min"] < lower_fence) | (df["wait_time_min"] > upper_fence)
].copy()

pct_outliers = len(df_outliers) / len(df) * 100
print(f"\n  Outliers: {len(df_outliers):,} rows ({pct_outliers:.1f}% of data)")

print("\n  Outliers by barista:")
print(
    df_outliers.groupby("agent")["wait_time_min"]
    .agg(count="count", mean="mean", max="max")
    .round(2)
    .sort_values("count", ascending=False)
    .to_string()
)

print("\n  Outliers by drink type:")
print(
    df_outliers.groupby("drink_name")["wait_time_min"]
    .agg(count="count", mean="mean", max="max")
    .round(2)
    .sort_values("count", ascending=False)
    .to_string()
)

df_outliers.to_csv(os.path.join(OUTPUT_DIR, "outliers.csv"), index=False)


# =============================================================================
# STEP 8 — VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Generating charts")
print("=" * 60)

overall_mean   = df["wait_time_min"].mean()
overall_median = df["wait_time_min"].median()

# ── Helper: add a mean reference line to an axis ─────────────────────────────
def add_mean_line(ax, value, label_prefix="Overall mean"):
    ax.axhline(value, color="red", linestyle="--", linewidth=1.4,
               label=f"{label_prefix}: {value:.1f} min")

# ────────────────────────────────────────────────────────────────────────────
# Chart 1: Distribution of wait times (histogram + density overlay)
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(df["wait_time_min"], bins=80, color="steelblue",
        edgecolor="white", alpha=0.85, label="Frequency")
ax.axvline(overall_mean,   color="red",    linestyle="--", linewidth=1.5,
           label=f"Mean: {overall_mean:.1f} min")
ax.axvline(overall_median, color="orange", linestyle="--", linewidth=1.5,
           label=f"Median: {overall_median:.1f} min")
ax.axvline(upper_fence,    color="darkred", linestyle=":",  linewidth=1.5,
           label=f"Outlier fence: {upper_fence:.1f} min")

ax.set_title("Distribution of Wait Times (Order Placed → Drink Completed)", fontsize=13)
ax.set_xlabel("Wait Time (minutes)")
ax.set_ylabel("Number of Orders")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "wait_time_distribution.png"), dpi=150)
plt.close()
print("  Saved: wait_time_distribution.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 2: Box plot — wait time by barista
# ────────────────────────────────────────────────────────────────────────────
# Sort agents by mean wait time (highest on top) for easy visual ranking.
agent_order = (
    df.groupby("agent")["wait_time_min"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(data=df, x="agent", y="wait_time_min",
            order=agent_order, hue="agent", palette="Blues_r",
            legend=False, ax=ax)
# Overlay individual points (alpha very low because n=10,000)
sns.stripplot(data=df.sample(min(1500, len(df)), random_state=42),
              x="agent", y="wait_time_min",
              order=agent_order, color="black", alpha=0.08, size=2.5, ax=ax)
add_mean_line(ax, overall_mean)
ax.set_title("Wait Time Distribution by Barista", fontsize=13)
ax.set_xlabel("Barista")
ax.set_ylabel("Wait Time (minutes)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "wait_time_by_agent.png"), dpi=150)
plt.close()
print("  Saved: wait_time_by_agent.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 3: Box plot — wait time by drink type
# ────────────────────────────────────────────────────────────────────────────
drink_order = (
    df.groupby("drink_name")["wait_time_min"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=df, x="drink_name", y="wait_time_min",
            order=drink_order, hue="drink_name", palette="Oranges_r",
            legend=False, ax=ax)
add_mean_line(ax, overall_mean)
ax.set_title("Wait Time Distribution by Drink Type", fontsize=13)
ax.set_xlabel("Drink Type")
ax.set_ylabel("Wait Time (minutes)")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "wait_time_by_drink.png"), dpi=150)
plt.close()
print("  Saved: wait_time_by_drink.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 4: Line chart — avg wait time and order volume by hour
# ────────────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()   # second y-axis for order count

ax1.plot(hourly_stats["hour_of_day"], hourly_stats["avg_wait_min"],
         marker="o", color="steelblue", linewidth=2, label="Avg wait (min)")
ax2.bar(hourly_stats["hour_of_day"], hourly_stats["order_count"],
        color="lightgray", alpha=0.5, label="Order count")

ax1.set_xlabel("Hour of Day")
ax1.set_ylabel("Avg Wait Time (minutes)", color="steelblue")
ax2.set_ylabel("Number of Orders", color="gray")
ax1.set_title("Wait Time and Order Volume by Hour of Day", fontsize=13)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "wait_time_by_hour.png"), dpi=150)
plt.close()
print("  Saved: wait_time_by_hour.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 5: Heatmap — avg wait time (agent × drink type)
# ────────────────────────────────────────────────────────────────────────────
pivot = df.pivot_table(
    values="wait_time_min", index="agent", columns="drink_name", aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Avg wait (min)"})
ax.set_title("Average Wait Time (min): Barista × Drink Type", fontsize=13)
ax.set_xlabel("Drink Type")
ax.set_ylabel("Barista")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "heatmap_agent_drink.png"), dpi=150)
plt.close()
print("  Saved: heatmap_agent_drink.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 6: Stacked bar — time decomposition by agent
#   Shows how much of each agent's total wait time is idle wait vs. prep time.
# ────────────────────────────────────────────────────────────────────────────
decomp = (
    df.groupby("agent")[["idle_wait_min", "prep_time_min"]]
    .mean()
    .sort_values("idle_wait_min", ascending=False)
)

fig, ax = plt.subplots(figsize=(9, 5))
decomp.plot(kind="bar", stacked=True, ax=ax,
            color=["#5B9BD5", "#ED7D31"], edgecolor="black")
ax.set_title("Avg Wait Time Decomposition by Barista\n(Idle Wait = queue time before barista starts  |  Prep = actual making time)", fontsize=11)
ax.set_xlabel("Barista")
ax.set_ylabel("Minutes")
ax.legend(["Idle wait (order placed → barista starts)",
           "Prep time (barista starts → drink ready)"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "time_decomposition_by_agent.png"), dpi=150)
plt.close()
print("  Saved: time_decomposition_by_agent.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 7: Send-back rate by barista
# ────────────────────────────────────────────────────────────────────────────
sb = agent_stats.set_index("agent")["send_back_pct"].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(sb.index, sb.values, color="salmon", edgecolor="black")
ax.axhline(overall_rate, color="red", linestyle="--", linewidth=1.4,
           label=f"Overall rate: {overall_rate:.1f}%")
ax.set_title("Order Send-Back Rate by Barista", fontsize=13)
ax.set_xlabel("Barista")
ax.set_ylabel("Send-Back Rate (%)")
ax.legend()
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "send_back_rate_by_agent.png"), dpi=150)
plt.close()
print("  Saved: send_back_rate_by_agent.png")

print(f"\n✓ Script 02 complete. All outputs saved to '{OUTPUT_DIR}/'.")
print("  Run 03_statistical_analysis.py next.")
