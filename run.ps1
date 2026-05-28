# run.ps1 — create venv, install deps, and launch the Streamlit app
param(
    [switch]$Reinstall
)

$venv = Join-Path $PSScriptRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venv
}

if ($Reinstall -or -not (Test-Path $python)) {
    Write-Host "Installing dependencies into virtual environment..."
    & $python -m pip install --upgrade pip
    & $python -m pip install -r "$(Join-Path $PSScriptRoot 'requirements.txt')"
}

Write-Host "Running full pipeline (clean → stats → dashboard)..."
# 00_run_all.py runs 01_clean_data.py, 02_descriptive_stats.py, and
# 03_statistical_analysis.py in order, then launches the Streamlit dashboard.
# Pass --data-only to skip the stat-export scripts when you only need a
# data refresh and the dashboard is already running (see update_data.ps1).
& $python "$(Join-Path $PSScriptRoot '00_run_all.py')"
