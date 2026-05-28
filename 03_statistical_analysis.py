# =============================================================================
# 03_statistical_analysis.py
# PURPOSE: Determine whether barista identity and/or drink type significantly
#          affect wait time, and pinpoint which specific groups are responsible.
#
# ANALYSES PERFORMED:
#   1. One-way ANOVA  — agent effect on wait_time_min
#   2. One-way ANOVA  — drink_name effect on wait_time_min
#   3. Tukey HSD      — pairwise barista comparisons
#   4. Tukey HSD      — pairwise drink comparisons
#   5. Tukey HSD      — visualized as significance heatmaps
#   6. Multiple OLS regression — agent + drink_name + hour_of_day +
#                                order_sent_back → wait_time_min
#   7. Supplemental: ANOVA on prep_time_min alone (isolates barista skill)
#
# DEPENDS ON: coffee_shop.db  (created by 01_clean_data.py)
# OUTPUTS:
#   outputs/tukey_agent.csv
#   outputs/tukey_drink.csv
#   outputs/regression_results.txt
#   outputs/anova_summary.txt
#   outputs/figures/  (several .png charts)
# =============================================================================

import sys
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations
import statsmodels.formula.api as smf
import os

sys.stdout.reconfigure(encoding="utf-8")

# ── Output directories ────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
FIG_DIR    = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

ALPHA  = 0.05      # significance threshold used throughout
DB_FILE = "coffee_shop.db"


# =============================================================================
# STEP 1 — LOAD DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading data from SQLite")
print("=" * 60)

conn = sqlite3.connect(DB_FILE)
df   = pd.read_sql_query("SELECT * FROM orders", conn)
conn.close()

print(f"  Rows loaded: {len(df):,}")

overall_mean = df["wait_time_min"].mean()


# =============================================================================
# HELPER: print a section header and save ANOVA results to a text buffer
# =============================================================================

anova_lines = []   # collects all ANOVA output for the summary text file

def anova_report(title, f_stat, p_val, eta_sq, n_groups):
    """Print and record a formatted ANOVA result block."""
    sig = "SIGNIFICANT" if p_val < ALPHA else "NOT SIGNIFICANT"
    block = [
        "",
        f"{'─'*60}",
        f"  {title}",
        f"{'─'*60}",
        f"  F-statistic  : {f_stat:.4f}",
        f"  p-value      : {p_val:.6f}",
        f"  Significance : {sig} (α = {ALPHA})",
        f"  η² (eta²)   : {eta_sq:.4f}  "
        f"({eta_sq*100:.1f}% of wait-time variance explained)",
        f"  Number of groups: {n_groups}",
    ]
    for line in block:
        print(line)
    anova_lines.extend(block)


# =============================================================================
# STEP 2 — ONE-WAY ANOVA: AGENT EFFECT ON WAIT TIME
# =============================================================================
# H₀: The mean wait time is equal across all baristas.
# H₁: At least one barista has a significantly different mean wait time.
#
# We use scipy.stats.f_oneway which takes each group as a separate array.

print("\n" + "=" * 60)
print("STEP 2: One-Way ANOVA — Barista effect on wait_time_min")
print("=" * 60)

groups_agent = [
    grp["wait_time_min"].values
    for _, grp in df.groupby("agent")
]

f_agent, p_agent = stats.f_oneway(*groups_agent)

# Eta-squared = SS_between / SS_total
# Measures the proportion of total variance in wait_time_min attributable to
# which barista served the order.
grand_mean   = df["wait_time_min"].mean()
ss_total     = ((df["wait_time_min"] - grand_mean) ** 2).sum()
ss_between_a = sum(
    len(g) * (g["wait_time_min"].mean() - grand_mean) ** 2
    for _, g in df.groupby("agent")
)
eta_sq_agent = ss_between_a / ss_total

anova_report("ANOVA: Barista → wait_time_min",
             f_agent, p_agent, eta_sq_agent, df["agent"].nunique())


# =============================================================================
# STEP 3 — ONE-WAY ANOVA: DRINK TYPE EFFECT ON WAIT TIME
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: One-Way ANOVA — Drink type effect on wait_time_min")
print("=" * 60)

groups_drink = [
    grp["wait_time_min"].values
    for _, grp in df.groupby("drink_name")
]

f_drink, p_drink = stats.f_oneway(*groups_drink)

ss_between_d = sum(
    len(g) * (g["wait_time_min"].mean() - grand_mean) ** 2
    for _, g in df.groupby("drink_name")
)
eta_sq_drink = ss_between_d / ss_total

anova_report("ANOVA: Drink type → wait_time_min",
             f_drink, p_drink, eta_sq_drink, df["drink_name"].nunique())


# =============================================================================
# STEP 4 — SUPPLEMENTAL ANOVA: AGENT EFFECT ON PREP TIME ONLY
# =============================================================================
# prep_time_min is the time from "barista starts" → "drink ready".
# This isolates the barista's personal speed from queue dynamics.
# A significant result here means some baristas genuinely make drinks faster.

print("\n" + "=" * 60)
print("STEP 4: Supplemental ANOVA — Barista effect on prep_time_min")
print("=" * 60)

groups_prep = [
    grp["prep_time_min"].values
    for _, grp in df.groupby("agent")
]

f_prep, p_prep = stats.f_oneway(*groups_prep)

grand_mean_prep = df["prep_time_min"].mean()
ss_total_prep   = ((df["prep_time_min"] - grand_mean_prep) ** 2).sum()
ss_between_prep = sum(
    len(g) * (g["prep_time_min"].mean() - grand_mean_prep) ** 2
    for _, g in df.groupby("agent")
)
eta_sq_prep = ss_between_prep / ss_total_prep

anova_report("ANOVA: Barista → prep_time_min (barista speed only)",
             f_prep, p_prep, eta_sq_prep, df["agent"].nunique())


# ── Save ANOVA summary to file ────────────────────────────────────────────────
with open(os.path.join(OUTPUT_DIR, "anova_summary.txt"), "w", encoding="utf-8") as f:
    f.write("ANOVA RESULTS SUMMARY\n")
    f.write("\n".join(anova_lines))
print(f"\n  ANOVA results saved to {OUTPUT_DIR}/anova_summary.txt")


# =============================================================================
# STEP 5 — TUKEY HSD PAIRWISE COMPARISONS: BARISTAS
# =============================================================================
# ANOVA tells us IF groups differ; Tukey HSD tells us WHICH pairs differ.
# Tukey's HSD controls the family-wise error rate (FWER) at α=0.05,
# accounting for the inflated false-positive risk from multiple comparisons.

print("\n" + "=" * 60)
print("STEP 5: Tukey HSD — pairwise barista comparisons")
print("=" * 60)

tukey_agent = pairwise_tukeyhsd(
    endog=df["wait_time_min"],
    groups=df["agent"],
    alpha=ALPHA,
)

print(tukey_agent)

# Build a tidy DataFrame from the Tukey result internals.
# statsmodels generates pairs in the same order as itertools.combinations,
# i.e. (g0,g1), (g0,g2), ..., (g1,g2), ... for sorted group labels.
groups_a  = list(tukey_agent.groupsunique)
pair_idx_a = list(combinations(range(len(groups_a)), 2))

tukey_agent_df = pd.DataFrame({
    "group1"    : [str(groups_a[i]) for i, j in pair_idx_a],
    "group2"    : [str(groups_a[j]) for i, j in pair_idx_a],
    "meandiff"  : tukey_agent.meandiffs.round(3),
    "p_adj"     : tukey_agent.pvalues.round(6),
    "lower_ci"  : tukey_agent.confint[:, 0].round(3),
    "upper_ci"  : tukey_agent.confint[:, 1].round(3),
    "reject_H0" : tukey_agent.reject,   # True = significantly different
})

print("\n  Tidy Tukey result (agent):")
print(tukey_agent_df.to_string(index=False))
tukey_agent_df.to_csv(os.path.join(OUTPUT_DIR, "tukey_agent.csv"), index=False)


# =============================================================================
# STEP 6 — TUKEY HSD PAIRWISE COMPARISONS: DRINK TYPES
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Tukey HSD — pairwise drink-type comparisons")
print("=" * 60)

tukey_drink = pairwise_tukeyhsd(
    endog=df["wait_time_min"],
    groups=df["drink_name"],
    alpha=ALPHA,
)

print(tukey_drink)

groups_d   = list(tukey_drink.groupsunique)
pair_idx_d = list(combinations(range(len(groups_d)), 2))

tukey_drink_df = pd.DataFrame({
    "group1"    : [str(groups_d[i]) for i, j in pair_idx_d],
    "group2"    : [str(groups_d[j]) for i, j in pair_idx_d],
    "meandiff"  : tukey_drink.meandiffs.round(3),
    "p_adj"     : tukey_drink.pvalues.round(6),
    "lower_ci"  : tukey_drink.confint[:, 0].round(3),
    "upper_ci"  : tukey_drink.confint[:, 1].round(3),
    "reject_H0" : tukey_drink.reject,
})

print("\n  Tidy Tukey result (drink):")
print(tukey_drink_df.to_string(index=False))
tukey_drink_df.to_csv(os.path.join(OUTPUT_DIR, "tukey_drink.csv"), index=False)


# =============================================================================
# STEP 7 — MULTIPLE LINEAR REGRESSION
# =============================================================================
# Model:
#   wait_time_min ~ C(agent) + C(drink_name) + hour_of_day + order_sent_back
#
# C() instructs statsmodels to treat agent and drink_name as categorical,
# dummy-coding them automatically. The first level alphabetically becomes the
# reference category (Alice for agent; Americano for drink_name).
#
# Interpretation of coefficients:
#   - C(agent)[T.Bob]: Bob's mean wait time is X min MORE than Alice's,
#     holding drink type, hour, and send-back constant.
#   - hour_of_day: each additional hour increases/decreases wait by X min.
#   - order_sent_back: sending an order back adds X min to wait time on average.

print("\n" + "=" * 60)
print("STEP 7: Multiple Linear Regression")
print("=" * 60)

formula = (
    "wait_time_min ~ "
    "C(agent) + C(drink_name) + hour_of_day + order_sent_back"
)

model = smf.ols(formula=formula, data=df).fit()

print(model.summary())

# Save regression output
with open(os.path.join(OUTPUT_DIR, "regression_results.txt"), "w", encoding="utf-8") as f:
    f.write(model.summary().as_text())
print(f"\n  Regression results saved to {OUTPUT_DIR}/regression_results.txt")

# ── Key regression metrics ────────────────────────────────────────────────────
print(f"\n  R-squared       : {model.rsquared:.4f}")
print(f"  Adj. R-squared  : {model.rsquared_adj:.4f}")
print(f"  Model F p-value : {model.f_pvalue:.6f}")

# ── Extract and display significant coefficients ─────────────────────────────
coef_df = pd.DataFrame({
    "coef"    : model.params,
    "pvalue"  : model.pvalues,
    "lower"   : model.conf_int()[0],
    "upper"   : model.conf_int()[1],
}).reset_index().rename(columns={"index": "term"})

sig_coef = coef_df[
    (coef_df["pvalue"] < ALPHA) & (coef_df["term"] != "Intercept")
].sort_values("coef", ascending=False)

print(f"\n  Significant predictors (p < {ALPHA}):")
print(sig_coef[["term", "coef", "pvalue", "lower", "upper"]].round(4).to_string(index=False))


# =============================================================================
# STEP 8 — VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Generating statistical charts")
print("=" * 60)


# ────────────────────────────────────────────────────────────────────────────
# Chart 1: Mean wait time by barista with CI error bars
# ────────────────────────────────────────────────────────────────────────────
# 95% CI = mean ± 1.96 * (std / sqrt(n))
agent_ci = df.groupby("agent")["wait_time_min"].agg(
    mean="mean", std="std", n="count"
).assign(ci95=lambda x: 1.96 * x["std"] / np.sqrt(x["n"]))
agent_ci = agent_ci.sort_values("mean", ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(agent_ci.index, agent_ci["mean"],
       yerr=agent_ci["ci95"], capsize=6,
       color="steelblue", edgecolor="black", error_kw={"linewidth": 1.5})
ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.4,
           label=f"Overall mean: {overall_mean:.1f} min")
ax.set_title(f"Mean Wait Time by Barista ± 95% CI\n"
             f"(ANOVA: F={f_agent:.2f}, p={p_agent:.4f}, η²={eta_sq_agent:.3f})",
             fontsize=12)
ax.set_xlabel("Barista")
ax.set_ylabel("Mean Wait Time (minutes)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "anova_agent_means.png"), dpi=150)
plt.close()
print("  Saved: anova_agent_means.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 2: Mean wait time by drink type with CI error bars
# ────────────────────────────────────────────────────────────────────────────
drink_ci = df.groupby("drink_name")["wait_time_min"].agg(
    mean="mean", std="std", n="count"
).assign(ci95=lambda x: 1.96 * x["std"] / np.sqrt(x["n"]))
drink_ci = drink_ci.sort_values("mean", ascending=False)

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(drink_ci.index, drink_ci["mean"],
       yerr=drink_ci["ci95"], capsize=6,
       color="coral", edgecolor="black", error_kw={"linewidth": 1.5})
ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.4,
           label=f"Overall mean: {overall_mean:.1f} min")
ax.set_title(f"Mean Wait Time by Drink Type ± 95% CI\n"
             f"(ANOVA: F={f_drink:.2f}, p={p_drink:.4f}, η²={eta_sq_drink:.3f})",
             fontsize=12)
ax.set_xlabel("Drink Type")
ax.set_ylabel("Mean Wait Time (minutes)")
ax.legend()
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "anova_drink_means.png"), dpi=150)
plt.close()
print("  Saved: anova_drink_means.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 3: Tukey HSD significance heatmap — AGENTS
# ────────────────────────────────────────────────────────────────────────────
# Build a symmetric matrix of adjusted p-values.  Green = no significant
# difference; red = significantly different at α=0.05.

def build_tukey_matrix(tukey_df, groups):
    """Create a symmetric n×n matrix of adjusted p-values from Tukey results."""
    n   = len(groups)
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for _, row in tukey_df.iterrows():
        mat.loc[row["group1"], row["group2"]] = row["p_adj"]
        mat.loc[row["group2"], row["group1"]] = row["p_adj"]
    # Diagonal = 1.0 (no difference with itself)
    for g in groups:
        mat.loc[g, g] = 1.0
    return mat

agents = sorted(df["agent"].unique())
tukey_mat_agent = build_tukey_matrix(tukey_agent_df, agents)

fig, ax = plt.subplots(figsize=(7, 5))
cmap = sns.diverging_palette(10, 130, as_cmap=True)   # red → green
sns.heatmap(tukey_mat_agent, annot=True, fmt=".3f",
            cmap=cmap, vmin=0, vmax=1,
            linewidths=1, ax=ax, cbar_kws={"label": "Adjusted p-value"})
ax.set_title(
    "Tukey HSD: Barista Pairwise Adjusted p-values\n"
    "(Red < 0.05 = significantly different; Green ≥ 0.05 = not significant)",
    fontsize=11,
)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tukey_heatmap_agent.png"), dpi=150)
plt.close()
print("  Saved: tukey_heatmap_agent.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 4: Tukey HSD significance heatmap — DRINK TYPES
# ────────────────────────────────────────────────────────────────────────────
drinks = sorted(df["drink_name"].unique())
tukey_mat_drink = build_tukey_matrix(tukey_drink_df, drinks)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(tukey_mat_drink, annot=True, fmt=".3f",
            cmap=cmap, vmin=0, vmax=1,
            linewidths=1, ax=ax, cbar_kws={"label": "Adjusted p-value"})
ax.set_title(
    "Tukey HSD: Drink Type Pairwise Adjusted p-values\n"
    "(Red < 0.05 = significantly different; Green ≥ 0.05 = not significant)",
    fontsize=11,
)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "tukey_heatmap_drink.png"), dpi=150)
plt.close()
print("  Saved: tukey_heatmap_drink.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 5: Regression coefficient plot
# ────────────────────────────────────────────────────────────────────────────
# Shows each predictor's estimated effect on wait_time_min with 95% CI.
# Significant terms (p<0.05) are shown in blue; non-significant in gray.
# The reference categories (Alice, Americano) are not shown—their effects
# are absorbed into the intercept.

coef_plot = coef_df[coef_df["term"] != "Intercept"].copy()
coef_plot["significant"] = coef_plot["pvalue"] < ALPHA
coef_plot = coef_plot.sort_values("coef")

colors    = ["steelblue" if sig else "lightgray"
             for sig in coef_plot["significant"]]
err_lower = coef_plot["coef"] - coef_plot["lower"]
err_upper = coef_plot["upper"] - coef_plot["coef"]

fig, ax = plt.subplots(figsize=(10, max(6, len(coef_plot) * 0.35)))
ax.errorbar(
    x=coef_plot["coef"],
    y=range(len(coef_plot)),
    xerr=[err_lower, err_upper],
    fmt="none",
    color="black",
    capsize=4,
    linewidth=1,
)
ax.scatter(coef_plot["coef"], range(len(coef_plot)),
           color=colors, s=60, zorder=5)
ax.axvline(0, color="black", linewidth=1)
ax.set_yticks(range(len(coef_plot)))
ax.set_yticklabels(coef_plot["term"], fontsize=8)
ax.set_xlabel("Effect on Wait Time (minutes relative to reference)")
ax.set_title(
    "Regression Coefficients ± 95% CI\n"
    "(Reference: Alice + Americano | Blue = significant p<0.05)",
    fontsize=11,
)

blue_patch = mpatches.Patch(color="steelblue", label="Significant (p<0.05)")
gray_patch = mpatches.Patch(color="lightgray",  label="Not significant")
ax.legend(handles=[blue_patch, gray_patch], loc="lower right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "regression_coefficients.png"), dpi=150)
plt.close()
print("  Saved: regression_coefficients.png")

# ────────────────────────────────────────────────────────────────────────────
# Chart 6: Violin plot — wait time by barista
#   Violins show the full distribution shape, making skew and bimodality
#   visible in a way a simple bar chart cannot.
# ────────────────────────────────────────────────────────────────────────────
agent_order_asc = (
    df.groupby("agent")["wait_time_min"]
    .mean()
    .sort_values()
    .index.tolist()
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x="agent", y="wait_time_min",
               order=agent_order_asc, palette="Blues", inner="quartile", ax=ax)
ax.axhline(overall_mean, color="red", linestyle="--", linewidth=1.4,
           label=f"Overall mean: {overall_mean:.1f} min")
ax.set_title("Wait Time Distribution by Barista (Violin)", fontsize=13)
ax.set_xlabel("Barista (sorted by mean)")
ax.set_ylabel("Wait Time (minutes)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "violin_wait_by_agent.png"), dpi=150)
plt.close()
print("  Saved: violin_wait_by_agent.png")

print(f"\n✓ Script 03 complete. All outputs saved to '{OUTPUT_DIR}/'.")
print("  Run 04_dashboard.py to launch the interactive Streamlit dashboard.")
