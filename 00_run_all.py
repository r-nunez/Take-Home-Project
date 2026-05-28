# =============================================================================
# run_all.py
# PURPOSE: Single entry point for the entire coffee shop analysis project.
#
# HOW TO RUN:
#   python run_all.py
#
# WHAT IT DOES:
#   1. Cleans the raw Excel data and saves it to coffee_shop.db
#   2. Generates descriptive statistics, outlier analysis, and charts
#   3. Runs ANOVA, Tukey HSD pairwise tests, and regression analysis
#   4. Launches the interactive Streamlit dashboard in your browser
#
# REQUIREMENT: coffee_shop_activity.xlsx must be in the same folder as this script.
# =============================================================================

import subprocess
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# All scripts live in the same directory as this file
HERE = Path(__file__).parent.resolve()

STEPS = [
    ("01_clean_data.py",         "Step 1/3 — Cleaning data & building database"),
    ("02_descriptive_stats.py",  "Step 2/3 — Computing descriptive statistics & charts"),
    ("03_statistical_analysis.py","Step 3/3 — Running ANOVA, Tukey HSD & regression"),
]


def run_script(filename, label):
    """
    Run a Python script as a subprocess, streaming its output live.
    Returns True on success, False on failure.
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    script_path = HERE / filename

    # Use the same Python interpreter that is running this script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(HERE),         # run from the project folder so relative paths work
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # ensure UTF-8 output
    )

    if result.returncode != 0:
        print(f"\n  ERROR: {filename} failed (exit code {result.returncode}).")
        print("  Fix the error above before proceeding.")
        return False

    return True


def launch_dashboard():
    """Launch the Streamlit dashboard and open it in the default browser."""
    dashboard_path = HERE / "04_dashboard.py"

    print(f"\n{'='*60}")
    print("  Launching Streamlit Dashboard")
    print(f"{'='*60}")
    print("  The dashboard will open in your default browser.")
    print("  Press Ctrl+C in this terminal to stop it.\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--server.headless", "false"],
        cwd=str(HERE),
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


if __name__ == "__main__":
    print("☕ Coffee Shop Operations Analysis — Full Pipeline")
    print("=" * 60)

    # Verify the raw data file is present before starting
    excel_file = HERE / "coffee_shop_activity.xlsx"
    if not excel_file.exists():
        print(f"\n  ERROR: '{excel_file.name}' not found in:\n  {HERE}")
        print("  Place the Excel file in the same folder as run_all.py and try again.")
        sys.exit(1)

    # Run each analysis script in order; stop immediately if any fails
    for filename, label in STEPS:
        success = run_script(filename, label)
        if not success:
            sys.exit(1)

    print(f"\n{'='*60}")
    print("  All analysis steps complete.")
    print(f"  Charts saved to: {HERE / 'outputs' / 'figures'}")
    print(f"  CSVs saved to  : {HERE / 'outputs'}")
    print(f"{'='*60}")

    # Launch the dashboard as the final step
    launch_dashboard()
