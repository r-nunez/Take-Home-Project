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

Write-Host "Launching Streamlit app..."
& $python -m streamlit run "$(Join-Path $PSScriptRoot '04_dashboard.py')"
