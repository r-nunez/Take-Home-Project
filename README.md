# Coffee Shop Operations Dashboard

This repository contains a Streamlit dashboard to monitor coffee shop barista performance, drink metrics, and statistical analyses. The primary app is `04_dashboard.py`.

## Quick start

1. Create a virtual environment and activate it (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# or
.\.venv\Scripts\activate.bat   # cmd
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app (two options):

- Use Streamlit directly (if `streamlit` is on your PATH):

```powershell
streamlit run 04_dashboard.py
```

- Or run via Python (the repository includes a fallback so this works even when the `streamlit` CLI isn't on PATH):

```powershell
python 04_dashboard.py
```

4. Open the URL shown in the terminal (usually `http://localhost:8501`).

## Files the app depends on
- `coffee_shop.db` — SQLite DB created by `01_clean_data.py`. Place it in the project root.
- `outputs/` — CSVs and figures from the analysis scripts. The app will create this folder if missing.

## Suppress Streamlit telemetry / onboarding prompt
To avoid the initial Streamlit onboarding prompt, this repo includes a local config at `.streamlit/config.toml`. If you want to apply this globally instead, create the following file at `%USERPROFILE%/.streamlit/config.toml`:

```toml
[browser]
gatherUsageStats = false
```

## Notes
- The script attempts to be robust to working directory differences by resolving paths relative to `04_dashboard.py`.
- If you see errors about missing tables or database, re-run `01_clean_data.py` to recreate `coffee_shop.db`.

If you want, I can also add a tiny `Makefile` or PowerShell script to automate environment setup and launch.

## Launch scripts

This repo includes small launcher scripts to automate environment setup and app startup:

- `run.ps1` — Windows PowerShell launcher: creates a `.venv`, installs `requirements.txt`, and launches the app. Use on Windows:

```powershell
.\run.ps1
# or force reinstall
.\run.ps1 -Reinstall
```

- `run.sh` — POSIX launcher for macOS/Linux/WSL: creates a `.venv`, installs `requirements.txt`, and launches the app. Use on POSIX systems:

```bash
chmod +x run.sh
./run.sh
./run.sh -r   # force reinstall
```

Both scripts call `python -m streamlit run 04_dashboard.py` so you don't need the `streamlit` CLI on your PATH.
