# update_data.ps1
# PURPOSE: Lightweight data refresh — re-ingests source files and updates
#          coffee_shop.db without running the stat-export scripts or relaunching
#          the dashboard.
#
# WHEN TO USE THIS:
#   The dashboard is already running in another terminal and you have new data.
#   After this script finishes, click "Refresh Data" in the dashboard sidebar
#   to reload from the updated database without restarting the server.
#
# HOW TO ADD NEW DATA:
#   Option A (recommended) — multi-file folder:
#     1. Create a data/ folder in the project root (if it does not exist).
#     2. Drop any number of .xlsx files into data/.
#        Each file must use the same sheet name ("in") and column layout as
#        the original coffee_shop_activity.xlsx.
#     3. Run this script.  All files are merged and deduplicated automatically.
#
#   Option B — legacy single file:
#     1. Replace coffee_shop_activity.xlsx with the updated file.
#     2. Run this script.
#
# REQUIREMENTS:
#   run.ps1 must have been run at least once to create the virtual environment.

$venv   = Join-Path $PSScriptRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

# Guard: ensure the virtual environment has been created
if (-not (Test-Path $python)) {
    Write-Host ""
    Write-Host "  ERROR: Virtual environment not found at $venv"
    Write-Host "  Run run.ps1 first to create it, then re-run this script."
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  Coffee Shop Data Refresh"
Write-Host "============================================================"
Write-Host ""

# Run only the ingestion script.  01_clean_data.py handles multi-file merging,
# schema validation, cleaning, feature engineering, and saving to the database.
& $python "$(Join-Path $PSScriptRoot '01_clean_data.py')"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "  Data updated successfully."
    Write-Host ""
    Write-Host "  Next steps:"
    Write-Host "    - If the dashboard is running: click 'Refresh Data' in"
    Write-Host "      the sidebar to reload the updated database."
    Write-Host "    - If the dashboard is not running: run run.ps1 to launch"
    Write-Host "      the full pipeline and open the dashboard."
    Write-Host "============================================================"
} else {
    Write-Host ""
    Write-Host "  ERROR: Data ingestion failed. See the output above for details."
    exit 1
}
