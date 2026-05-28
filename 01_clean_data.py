# =============================================================================
# 01_clean_data.py
# PURPOSE: Load raw Excel data, clean it, engineer time-based features,
#          and persist the result to a local SQLite database.
#
# RUN FIRST: All downstream scripts (02, 03, 04) depend on coffee_shop.db.
#
# INPUTS:  coffee_shop_activity.xlsx
# OUTPUTS: coffee_shop.db  (SQLite database, table: "orders")
# =============================================================================

import sys
import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path

# Allow Unicode characters (arrows, checkmarks, etc.) in terminal output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# ── File paths ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"           # drop new .xlsx files here for multi-file ingestion
EXCEL_FILE = BASE_DIR / "coffee_shop_activity.xlsx"   # legacy single-file fallback
DB_FILE    = str(BASE_DIR / "coffee_shop.db")

# ── Required columns — checked before any processing ────────────────────────
EXPECTED_COLUMNS = {
    "order_id", "drink_name", "agent", "order_sent_back",
    "time_entered", "time_ordered", "time_making_started",
    "time_completed", "time_left",
}

# ── Timestamp column names (used repeatedly below) ──────────────────────────
TIME_COLS = [
    "time_entered",
    "time_ordered",
    "time_making_started",
    "time_completed",
    "time_left",
]


# =============================================================================
# STEP 1 — LOAD RAW DATA
# =============================================================================

print("=" * 60)
print("STEP 1: Loading raw data from Excel")
print("=" * 60)

# ── Multi-file ingestion ──────────────────────────────────────────────────────
# To add new data, drop one or more .xlsx files into the data/ folder.
# Each file must have the same sheet name ("in") and column layout as the
# original coffee_shop_activity.xlsx.  All files are merged into one DataFrame
# before any cleaning or feature engineering begins.
#
# If the data/ folder does not exist (or is empty), the script falls back to
# the legacy single-file path (coffee_shop_activity.xlsx in the project root),
# which preserves backward compatibility for existing setups.
if DATA_DIR.exists() and any(DATA_DIR.glob("*.xlsx")):
    source_files = sorted(DATA_DIR.glob("*.xlsx"))
    print(f"  data/ folder detected — loading {len(source_files)} file(s):")
    frames = []
    for f in source_files:
        try:
            frame = pd.read_excel(f, sheet_name="in")
            print(f"    {f.name}: {len(frame):,} rows")
            frames.append(frame)
        except Exception as e:
            # Surface the filename so the user knows exactly which file failed
            print(f"  ERROR loading {f.name}: {e}")
            sys.exit(1)
    # Stack all files vertically; reset_index keeps row numbers contiguous
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows loaded from all files: {len(df):,}")
else:
    # Legacy path: single master Excel file in the project root
    if not EXCEL_FILE.exists():
        print(f"\n  ERROR: No data files found.")
        print(f"  Either place '{EXCEL_FILE.name}' in the project root,")
        print(f"  or create a 'data/' folder and add .xlsx files there.")
        sys.exit(1)
    print(f"  Loading {EXCEL_FILE.name}")
    df = pd.read_excel(str(EXCEL_FILE), sheet_name="in")
    print(f"  Rows loaded: {len(df):,}")

# ── Schema validation ─────────────────────────────────────────────────────────
# Check for required columns before doing any work.  If a file has been
# exported with renamed or missing columns, this exits immediately with a clear
# error message rather than failing silently mid-pipeline.
missing_cols = EXPECTED_COLUMNS - set(df.columns)
if missing_cols:
    print(f"\n  ERROR: Missing required columns: {sorted(missing_cols)}")
    print(f"  Columns present in file: {sorted(df.columns.tolist())}")
    sys.exit(1)
print(f"  Schema check passed — all required columns present.")

print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")


# =============================================================================
# STEP 2 — DATA CLEANING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: Cleaning data")
print("=" * 60)

rows_start = len(df)

# ── 2a. Drop rows with any missing timestamp ─────────────────────────────────
# Timestamps are essential for every KPI; rows without them are unanalyzable.
df = df.dropna(subset=TIME_COLS)
print(f"  Dropped {rows_start - len(df):,} rows with missing timestamps.")

# ── 2b. Drop rows with missing categorical fields ────────────────────────────
rows_before = len(df)
df = df.dropna(subset=["drink_name", "agent"])
print(f"  Dropped {rows_before - len(df):,} rows with missing drink/agent.")

# ── 2c. Coerce timestamp columns to datetime ─────────────────────────────────
for col in TIME_COLS:
    df[col] = pd.to_datetime(df[col])

# ── 2d. Standardize text columns (strip stray whitespace, consistent casing) ─
df["drink_name"] = df["drink_name"].str.strip()
df["agent"]      = df["agent"].str.strip()

# ── 2e. Fill missing order_sent_back with 0 and cast to int ──────────────────
# Missing means the order was NOT sent back; treat as 0.
df["order_sent_back"] = df["order_sent_back"].fillna(0).astype(int)

print(f"  Rows remaining after cleaning: {len(df):,}")


# =============================================================================
# STEP 3 — LOGICAL VALIDATION (timestamp sequence check)
# =============================================================================
# Expected order for every order:
#   time_entered ≤ time_ordered ≤ time_making_started ≤ time_completed ≤ time_left
#
# Any row that violates this sequence has corrupt timestamps and is removed.

print("\n" + "=" * 60)
print("STEP 3: Logical validation — timestamp sequence")
print("=" * 60)

rows_before = len(df)

invalid_mask = (
    (df["time_ordered"]         < df["time_entered"])         |
    (df["time_making_started"]  < df["time_ordered"])         |
    (df["time_completed"]       < df["time_making_started"])  |
    (df["time_left"]            < df["time_completed"])
)

df_invalid = df[invalid_mask].copy()   # save for inspection if needed
df         = df[~invalid_mask].copy()

print(f"  Rows with bad timestamp order: {len(df_invalid):,}")
print(f"  Rows remaining after validation: {len(df):,}")

if len(df_invalid) > 0:
    print("  Sample of removed rows:")
    print(df_invalid[["order_id"] + TIME_COLS].head(3).to_string())


# =============================================================================
# STEP 4 — FEATURE ENGINEERING
# =============================================================================
# We decompose the customer journey into distinct time segments.
# All durations are expressed in MINUTES for readability.
#
# Journey timeline:
#   [enter shop] → [place order] → [barista starts] → [drink ready] → [leave]
#       queue_wait      idle_wait         prep_time         dwell_time
#   |←────────────── wait_time_min (PRIMARY KPI) ──────────────→|
#                   (time_ordered → time_completed)

print("\n" + "=" * 60)
print("STEP 4: Engineering features")
print("=" * 60)

def seconds_to_minutes(timedelta_series):
    """Convert a pandas timedelta Series to float minutes."""
    return timedelta_series.dt.total_seconds() / 60

# ── PRIMARY KPI ───────────────────────────────────────────────────────────────
# wait_time_min: time from order placed → drink handed to customer
df["wait_time_min"] = seconds_to_minutes(df["time_completed"] - df["time_ordered"])

# ── DIAGNOSTIC SEGMENTS ───────────────────────────────────────────────────────
# queue_wait_min : time between entering the shop and placing the order
df["queue_wait_min"] = seconds_to_minutes(df["time_ordered"]        - df["time_entered"])

# idle_wait_min  : time between order placed and barista picking it up
df["idle_wait_min"]  = seconds_to_minutes(df["time_making_started"] - df["time_ordered"])

# prep_time_min  : time the barista actually spends making the drink
df["prep_time_min"]  = seconds_to_minutes(df["time_completed"]      - df["time_making_started"])

# dwell_time_min : how long the customer lingers after receiving the drink
df["dwell_time_min"] = seconds_to_minutes(df["time_left"]           - df["time_completed"])

# ── TEMPORAL CONTEXT ──────────────────────────────────────────────────────────
# Useful for time-of-day and day-of-week breakdowns in the dashboard.
df["hour_of_day"] = df["time_ordered"].dt.hour
df["day_of_week"] = df["time_ordered"].dt.day_name()
df["date"]        = df["time_ordered"].dt.date.astype(str)   # stored as string in SQLite

# ── SUMMARY OF NEW COLUMNS ────────────────────────────────────────────────────
new_cols = [
    "wait_time_min", "queue_wait_min", "idle_wait_min",
    "prep_time_min", "dwell_time_min",
]
print("  New time-metric columns (in minutes):")
for col in new_cols:
    print(f"    {col:<20} | mean={df[col].mean():6.2f} | "
          f"min={df[col].min():6.2f} | max={df[col].max():6.2f}")

print(f"\n  Temporal columns: hour_of_day, day_of_week, date")


# =============================================================================
# STEP 5 — SAVE TO SQLITE
# =============================================================================
# We convert datetime columns to strings for SQLite compatibility
# (SQLite has no native datetime type; ISO-format strings are standard).

print("\n" + "=" * 60)
print("STEP 5: Saving to SQLite")
print("=" * 60)

df_to_save = df.copy()
for col in TIME_COLS:
    df_to_save[col] = df_to_save[col].astype(str)

# ── Cross-file deduplication ──────────────────────────────────────────────────
# When multiple files are loaded from data/, the same order might appear in
# more than one export (e.g., a record near the end of January's file also
# appears at the start of February's file).  Keeping only the first occurrence
# of each order_id ensures the database never double-counts an order.
rows_before_dedup = len(df_to_save)
df_to_save = df_to_save.drop_duplicates(subset=["order_id"], keep="first")
n_dupes = rows_before_dedup - len(df_to_save)
if n_dupes:
    print(f"  Removed {n_dupes:,} duplicate order_id rows (overlap between files).")

conn = sqlite3.connect(DB_FILE)
# if_exists="replace" drops and recreates the table on every run, so the
# database always reflects the exact contents of the current source files.
df_to_save.to_sql("orders", conn, if_exists="replace", index=False)

# ── Quick verification query via SQL ─────────────────────────────────────────
verification_sql = """
    SELECT
        agent,
        COUNT(*)                                        AS total_orders,
        ROUND(AVG(wait_time_min), 2)                    AS avg_wait_min,
        ROUND(MIN(wait_time_min), 2)                    AS min_wait_min,
        ROUND(MAX(wait_time_min), 2)                    AS max_wait_min,
        ROUND(SUM(order_sent_back) * 100.0 / COUNT(*), 2) AS send_back_pct
    FROM orders
    GROUP BY agent
    ORDER BY avg_wait_min DESC;
"""
print("\n  SQL verification — avg wait time by barista:")
result = pd.read_sql_query(verification_sql, conn)
print(result.to_string(index=False))

conn.close()

print(f"\n  Saved {len(df_to_save):,} rows → table 'orders' in '{DB_FILE}'")
print("\n✓ Script 01 complete. Run 02_descriptive_stats.py next.")
