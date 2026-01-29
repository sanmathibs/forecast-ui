# Forecast UI

Standalone Streamlit experience for sales & footfall forecasting, accuracy proof vs Mecca baseline, and performance diagnostics.

## Run locally
```
cd forecast-ui
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py --server.port 8501
```
Login password is set via `FORECAST_APP_PASSWORD` (default `demo`). Optional `FORECAST_APP_USER` sets the default username.

## Deploy to Streamlit Cloud
- Repo main file: `ui/app.py`
- Requirements: `requirements.txt`
- Env vars: `FORECAST_APP_PASSWORD` (required), `FORECAST_APP_USER` (optional)

## Data
Expects `Sales Labour Footfall for 12 stores.xlsx` in the repo root. Path resolved via `api/core/config.py`.

## Contents
- `ui/` Streamlit UI (three tabs: Forecast, Historical Analysis, Performance, plus backtest)
- `api/` Forecast engine (QuantumForecast + helpers) used locally by the UI
- `.streamlit/config.toml` Theme config
- `requirements.txt` Dependencies

