#!/usr/bin/env bash
# run.sh — cross-platform (POSIX) launcher for the Streamlit app
# Usage:
#   ./run.sh          # create venv if missing, install deps, run app
#   ./run.sh -r       # force reinstall dependencies

set -euo pipefail
REINSTALL=0
while getopts "r" opt; do
  case ${opt} in
    r) REINSTALL=1 ;;
    *) echo "Usage: $0 [-r]"; exit 1 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PY_BIN="$VENV_DIR/bin/python"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR..."
  python3 -m venv "$VENV_DIR"
fi

if [ $REINSTALL -eq 1 ] || [ ! -x "$PY_BIN" ]; then
  echo "Installing dependencies into virtual environment..."
  "$PY_BIN" -m pip install --upgrade pip
  "$PY_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

echo "Launching Streamlit app..."
# Use python -m streamlit to avoid requiring streamlit binary on PATH
exec "$PY_BIN" -m streamlit run "$ROOT_DIR/04_dashboard.py"
