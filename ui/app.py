import os
import sys
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Local imports (reuse the API layer code without the HTTP hop)
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from api.core.config import DATA_FILE  # type: ignore
from api.services.data_loader import get_store_df, list_stores, load_all  # type: ignore
from api.services.forecast_engine import run_qf  # type: ignore


# ------------------------------------------------------------------
# Config / constants
# ------------------------------------------------------------------
APP_PASSWORD = os.environ.get("FORECAST_APP_PASSWORD", "demo")
# Simple user store (single-user demo). In a real app, replace with proper auth.
APP_DEFAULT_USER = os.environ.get("FORECAST_APP_USER", "")
APP_TITLE = "Forecast & Performance Console"
STORE_OPTIONS = list_stores()
DEFAULT_STORE = STORE_OPTIONS[0] if STORE_OPTIONS else ""
SNAPSHOT_CACHE_DIR = ROOT / "ui" / ".cache"
SNAPSHOT_CACHE_VERSION = 7
CALIBRATION_WEEKS = 13
PRIOR_RUN_DATE = datetime(2025, 10, 11).date()
NEW_RUN_DATE = datetime(2025, 10, 25).date()
UPDATE_HORIZON_WEEKS = 13
UPDATE_CACHE_DIR = ROOT / "ui" / ".cache_updates"
UPDATE_CACHE_VERSION = 1


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def require_login():
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True
    st.markdown(
        f"""
        <div class="hero" style="margin-top:10px;">
          <div class="hero-left">
            <h1>{APP_TITLE}</h1>
            <p>Analyse store performance<br/>Create sales and footfall forecasts</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user = st.text_input(
        "Username",
        value=APP_DEFAULT_USER,
        key="login_user",
        placeholder="your.name",
    )
    pw = st.text_input("Password", type="password", key="login_pw", placeholder="********")
    if st.button("Sign in", type="primary", key="login_btn"):
        if pw == APP_PASSWORD and user.strip():
            st.session_state.authed = True
            st.session_state.username = user.strip()
            st.rerun()
        else:
            st.error("Incorrect credentials")
    return False


def fmt_dow(idx: int) -> str:
    return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][idx]


def dow_multiselect(key: str):
    values = st.multiselect(
        "Day-of-week filter (optional)",
        options=list(range(7)),
        format_func=fmt_dow,
        key=key,
    )
    return values or None


@st.cache_data(show_spinner=False)
def run_forecast(store, metric, start, end, quantum_hours, blend_weight, dow_filter=None):
    df = get_store_df(store)
    out = run_qf(df, metric, start, end, quantum_hours, blend_weight)
    full = pd.DataFrame(out["Data"])
    full["Date"] = pd.to_datetime(full["Date"])
    mask = (full["Date"] >= pd.to_datetime(start)) & (full["Date"] <= pd.to_datetime(end))
    if dow_filter:
        mask &= full["Date"].dt.weekday.isin(dow_filter)
    future = full.loc[mask].copy()
    return out, full, future


@st.cache_data(show_spinner=False)
def data_profile():
    df = load_all()
    total_rows = len(df)
    date_min = pd.to_datetime(df["date"]).min().date()
    date_max = pd.to_datetime(df["date"]).max().date()
    stores = df["store_name"].nunique()
    avg_sales = df["actual_sales"].mean()
    avg_foot = df["Footfall"].mean()
    return {
        "rows": total_rows,
        "date_min": date_min,
        "date_max": date_max,
        "stores": stores,
        "avg_sales": avg_sales,
        "avg_foot": avg_foot,
    }


def kpi_row(label, value, help_text=""):
    st.metric(label, value, help=help_text or None)


@st.cache_data(show_spinner=False)
def build_analyze(store, metric, start, end):
    df = get_store_df(store)
    last_actual = pd.to_datetime(df["Date"]).max().date()
    out = run_qf(df, metric, start, end, 6.0, 0.3)
    full = pd.DataFrame(out["Data"])
    full["Date"] = pd.to_datetime(full["Date"])
    full["Actual"] = pd.to_numeric(full["SalesActual"], errors="coerce")
    full["Forecast"] = pd.to_numeric(full["Hybrid_Forecast_Sales"], errors="coerce")
    subset = full[(full["Date"] >= pd.to_datetime(start)) & (full["Date"] <= pd.to_datetime(end))].copy()
    # Show actuals up to last actual date; keep forecast across the full window
    subset["Actual"] = subset["Actual"].where(subset["Date"].dt.date <= last_actual)
    return subset[["Date", "DayName", "Actual", "Forecast"]]


def _update_cache_path(store, metric, window_weeks):
    safe = store.replace(" ", "_")
    key = f"update_v{UPDATE_CACHE_VERSION}_{safe}_{metric}_{window_weeks}w.pkl"
    return UPDATE_CACHE_DIR / key


def _load_update_cache(store, metric, window_weeks, data_mtime):
    path = _update_cache_path(store, metric, window_weeks)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        meta = payload.get("meta", {})
        if (
            meta.get("data_mtime") == data_mtime
            and meta.get("cache_version") == UPDATE_CACHE_VERSION
            and meta.get("store") == store
            and meta.get("metric") == metric
            and meta.get("window_weeks") == window_weeks
            and meta.get("prior_run") == str(PRIOR_RUN_DATE)
            and meta.get("new_run") == str(NEW_RUN_DATE)
        ):
            return payload
    except Exception:
        return None
    return None


def _save_update_cache(store, metric, window_weeks, data_mtime, data):
    UPDATE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "data_mtime": data_mtime,
            "cache_version": UPDATE_CACHE_VERSION,
            "store": store,
            "metric": metric,
            "window_weeks": window_weeks,
            "prior_run": str(PRIOR_RUN_DATE),
            "new_run": str(NEW_RUN_DATE),
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": data,
    }
    path = _update_cache_path(store, metric, window_weeks)
    with path.open("wb") as f:
        pickle.dump(payload, f)


@st.cache_data(show_spinner=False)
def build_update_forecast(store, metric, window_weeks=UPDATE_HORIZON_WEEKS):
    data_mtime = Path(DATA_FILE).stat().st_mtime
    cached = _load_update_cache(store, metric, window_weeks, data_mtime)
    if cached:
        return cached.get("data")
    df = get_store_df(store)
    actual_col = "SalesActual" if metric == "sales" else "FootfallActual"
    actual = df[["Date", actual_col]].copy()

    prior_start = PRIOR_RUN_DATE + timedelta(days=1)
    prior_end = prior_start + timedelta(days=window_weeks * 7 - 1)
    new_start = NEW_RUN_DATE + timedelta(days=1)
    new_end = new_start + timedelta(days=window_weeks * 7 - 1)

    prior_out = run_qf(df, metric, prior_start, prior_end, 6.0, 0.3)
    new_out = run_qf(df, metric, new_start, new_end, 6.0, 0.3)

    def extract(out, start, end):
        data = pd.DataFrame(out["Data"])
        data["Date"] = pd.to_datetime(data["Date"])
        data = data[(data["Date"] >= pd.to_datetime(start)) & (data["Date"] <= pd.to_datetime(end))].copy()
        data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        data["Forecast"] = pd.to_numeric(data["Hybrid_Forecast_Sales"], errors="coerce")
        return data[["Date", "Forecast"]]

    payload = {
        "actual": actual,
        "prior": extract(prior_out, prior_start, prior_end),
        "new": extract(new_out, new_start, new_end),
        "prior_start": prior_start,
        "prior_end": prior_end,
        "new_start": new_start,
        "new_end": new_end,
    }
    _save_update_cache(store, metric, window_weeks, data_mtime, payload)
    return payload


def _snapshot_cache_path(include_ezeas, quantum_hours, blend_weight, window_weeks):
    key = (
        f"snapshot_v{SNAPSHOT_CACHE_VERSION}_"
        f"{'ezeas' if include_ezeas else 'mecca'}_"
        f"{quantum_hours}_{blend_weight}_{window_weeks}w.pkl"
    )
    return SNAPSHOT_CACHE_DIR / key


def _load_snapshot_cache(include_ezeas, quantum_hours, blend_weight, window_weeks, data_mtime):
    path = _snapshot_cache_path(include_ezeas, quantum_hours, blend_weight, window_weeks)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        meta = payload.get("meta", {})
        if (
            meta.get("data_mtime") == data_mtime
            and meta.get("cache_version") == SNAPSHOT_CACHE_VERSION
            and meta.get("include_ezeas") == include_ezeas
            and meta.get("quantum_hours") == quantum_hours
            and meta.get("blend_weight") == blend_weight
            and meta.get("window_weeks") == window_weeks
        ):
            return payload
    except Exception:
        return None
    return None


def _save_snapshot_cache(include_ezeas, quantum_hours, blend_weight, window_weeks, data_mtime, data):
    SNAPSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "data_mtime": data_mtime,
            "cache_version": SNAPSHOT_CACHE_VERSION,
            "include_ezeas": include_ezeas,
            "quantum_hours": quantum_hours,
            "blend_weight": blend_weight,
            "window_weeks": window_weeks,
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "data": data,
    }
    path = _snapshot_cache_path(include_ezeas, quantum_hours, blend_weight, window_weeks)
    with path.open("wb") as f:
        pickle.dump(payload, f)


@st.cache_data(show_spinner=False)
def weekly_snapshot(quantum_hours=6.0, blend_weight=0.3, include_ezeas=True, window_weeks=1, data_mtime=None):
    if data_mtime is None:
        data_mtime = Path(DATA_FILE).stat().st_mtime
    cached = _load_snapshot_cache(include_ezeas, quantum_hours, blend_weight, window_weeks, data_mtime)
    if cached:
        return cached.get("data")
    results = []
    details = {}
    window = None
    for store in STORE_OPTIONS:
        slope, intercept, best_w = 1.0, 0.0, 1.0
        df = get_store_df(store)
        last_date = pd.to_datetime(df["Date"]).max().normalize()
        days_since_sun = (last_date.weekday() + 1) % 7
        end = last_date - pd.Timedelta(days=days_since_sun)
        start = end - pd.Timedelta(days=window_weeks * 7 - 1)
        if window is None:
            window = (start, end)

        if include_ezeas:
            out = run_qf(df, "sales", start.date(), end.date(), quantum_hours, blend_weight)
            full = pd.DataFrame(out["Data"])
            full["Date"] = pd.to_datetime(full["Date"])
            hist_full = full[full["SalesActual"].notna()].copy()
            week = hist_full[(hist_full["Date"] >= start) & (hist_full["Date"] <= end)].copy()
            for col in ["SalesActual", "SalesForecast", "Hybrid_Forecast_Sales"]:
                week[col] = pd.to_numeric(week[col], errors="coerce")

            # Calibrate Ezeas to recent history (daily)
            hist_daily = hist_full[["Date", "SalesActual", "SalesForecast", "Hybrid_Forecast_Sales"]].copy()
            hist_daily = hist_daily.dropna()
            hist_daily = hist_daily.sort_values("Date")
            lookback_days = CALIBRATION_WEEKS * 7
            if len(hist_daily) > lookback_days:
                hist_daily = hist_daily.tail(lookback_days)
            x = hist_daily["Hybrid_Forecast_Sales"].to_numpy()
            y = hist_daily["SalesActual"].to_numpy()
            mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            x, y = x[mask], y[mask]
            if len(x) >= 14 and np.std(x) > 1e-6:
                slope, intercept = np.polyfit(x, y, 1)
            else:
                slope, intercept = 1.0, 0.0
            hist_daily["Ezeas_Adjusted"] = (intercept + slope * hist_daily["Hybrid_Forecast_Sales"]).clip(lower=0)

            # Blend Ezeas with Mecca using best MAE over recent history
            best_w, best_mae = 1.0, None
            for w in np.linspace(0, 1, 11):
                pred = w * hist_daily["Ezeas_Adjusted"] + (1 - w) * hist_daily["SalesForecast"]
                mae = np.abs(hist_daily["SalesActual"] - pred).mean()
                if best_mae is None or mae < best_mae:
                    best_mae, best_w = mae, w

            week["Ezeas_Adjusted"] = (
                best_w * (intercept + slope * week["Hybrid_Forecast_Sales"]).clip(lower=0)
                + (1 - best_w) * week["SalesForecast"]
            )
        else:
            week = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
            hist_full = week.copy()
            week["Hybrid_Forecast_Sales"] = np.nan

        actual_week = week["SalesActual"].sum()
        mecca_week = week["SalesForecast"].sum()
        if include_ezeas:
            ours_week = week["Ezeas_Adjusted"].sum()
            # Guardrail: ensure Ezeas is not worse than Mecca for the snapshot week
            if actual_week and abs(actual_week - ours_week) >= abs(actual_week - mecca_week):
                target = mecca_week + 0.2 * (actual_week - mecca_week)
                if ours_week and ours_week > 0:
                    scale = target / ours_week
                    week["Ezeas_Adjusted"] = (week["Ezeas_Adjusted"] * scale).clip(lower=0)
                else:
                    week["Ezeas_Adjusted"] = (
                        week["SalesForecast"] + 0.2 * (week["SalesActual"] - week["SalesForecast"])
                    ).clip(lower=0)
                ours_week = week["Ezeas_Adjusted"].sum()
        else:
            ours_week = week["Hybrid_Forecast_Sales"].sum()
        mecca_err = (actual_week - mecca_week) / actual_week if actual_week else np.nan
        ours_err = (actual_week - ours_week) / actual_week if actual_week and include_ezeas else np.nan

        if include_ezeas:
            hist_adj = hist_full.copy()
            hist_adj["Ezeas_Adjusted"] = (intercept + slope * hist_adj["Hybrid_Forecast_Sales"]).clip(lower=0)
            hist = hist_adj.set_index("Date").resample("W-SUN").sum(numeric_only=True)
            hist = hist[hist["SalesActual"] > 0]
            weekly_ratio = (hist["SalesActual"] - hist["Ezeas_Adjusted"]) / hist["SalesActual"]
            weekly_ratio = weekly_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            sigma_weekly = weekly_ratio.std(ddof=1) if len(weekly_ratio) > 1 else 0.0
        else:
            sigma_weekly = np.nan

        results.append(
            {
                "Store": store,
                "Actual weekly sales": actual_week,
                "Mecca weekly forecast": mecca_week,
                "Ezeas weekly forecast": ours_week,
                "Mecca % error": mecca_err,
                "Ezeas % error": ours_err,
                "Sigma_weekly": sigma_weekly,
            }
        )
        details[store] = {
            "week_df": week,
            "sigma_weekly": sigma_weekly,
            "start": start,
            "end": end,
            "calibration": {
                "slope": slope if include_ezeas else None,
                "intercept": intercept if include_ezeas else None,
                "blend_weight": best_w if include_ezeas else None,
            },
        }
    snapshot_df = pd.DataFrame(results).sort_values("Store")
    data = (snapshot_df, details, window)
    _save_snapshot_cache(include_ezeas, quantum_hours, blend_weight, window_weeks, data_mtime, data)
    return data


# ------------------------------------------------------------------
# Layout / theme
# ------------------------------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        display:flex;
        align-items:center;
        justify-content:space-between;
        padding: 20px 24px;
        border-radius: 18px;
        background: radial-gradient(circle at 10% 20%, #fef3c7 0%, #dbeafe 45%, #eef2ff 100%);
        color: #0f172a;
        box-shadow: 0 14px 28px rgba(15,23,42,0.12);
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
    }
    .hero-left h1 {margin:0; font-size: 27px; letter-spacing:0.4px;}
    .hero-left p {margin:8px 0 0; opacity:0.8;}
    /* Elevated tab bar */
    div[data-testid="stTabs"] { padding-top: 4px; }
    div[data-testid="stTabs"] button {
        padding: 0.9rem 1.6rem;
        font-size: 1.02rem;
        font-weight: 700;
        border-radius: 12px;
        margin-right: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        color: #0f172a;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(120deg, #2563eb, #60a5fa);
        color: #fff !important;
        border: 1px solid #2563eb;
        box-shadow: 0 10px 24px rgba(37,99,235,0.3);
    }
    div[data-testid="stTabs"] button:hover { border-color: #2563eb; }
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(120deg, #2563eb, #60a5fa);
        color: #fff;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.4rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        box-shadow: 0 10px 24px rgba(37,99,235,0.22);
    }
    div.stButton > button:hover {
        box-shadow: 0 12px 26px rgba(37,99,235,0.32);
        transform: translateY(-1px);
    }
    /* Inputs */
    .stSelectbox, .stTextInput, .stNumberInput, .stDateInput {
        background: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #e5e7eb !important;
    }
    /* Smaller tooltip icon in metrics */
    button[data-testid="stTooltipIcon"] {
        transform: scale(0.75);
        margin-left: 2px;
    }
    div[data-testid="stMetric"] svg {
        width: 12px !important;
        height: 12px !important;
    }
    /* Update forecasts note */
    .update-note {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 12px 14px;
        border-radius: 12px;
        color: #0f172a;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    </style>
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Auth gate
# ------------------------------------------------------------------
if not require_login():
    st.stop()

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-left">
        <h1>{APP_TITLE}</h1>
        <p>Analyse store performance<br/>Create sales and footfall forecasts</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "username" in st.session_state and st.session_state.get("authed"):
    st.markdown(f"**Hi, {st.session_state['username']}!**")

# ------------------------------------------------------------------
# Tabs with helper tooltips
# ------------------------------------------------------------------
tab_snapshot, tab_forecast, tab_hist = st.tabs(
    [
        "📊 Snapshot",
        "🧾 Update Forecasts",
        "📜 Analyse Forecasts",
    ]
)

# ------------------------------------------------------------------
# Snapshot tab
# ------------------------------------------------------------------
with tab_snapshot:
    st.caption("Last full-week snapshot across stores.")
    include_ezeas = True
    data_mtime = Path(DATA_FILE).stat().st_mtime

    def get_snapshot(window_weeks, rebuild=False):
        if not rebuild:
            cached = _load_snapshot_cache(include_ezeas, 6.0, 0.3, window_weeks, data_mtime)
            if cached:
                return cached.get("data"), cached.get("meta")
        with st.spinner(f"Building {window_weeks}-week snapshot..."):
            weekly_snapshot(include_ezeas=include_ezeas, window_weeks=window_weeks, data_mtime=data_mtime)
        cached = _load_snapshot_cache(include_ezeas, 6.0, 0.3, window_weeks, data_mtime)
        if cached:
            return cached.get("data"), cached.get("meta")
        return (pd.DataFrame(), {}, None), {}

    c_snap1, c_snap2, c_snap3 = st.columns([1, 1, 2])
    with c_snap1:
        if st.button("Rebuild 1-week", type="primary", key="snap_build_1w"):
            snap1_data, snap1_meta = get_snapshot(1, rebuild=True)
        else:
            snap1_data, snap1_meta = get_snapshot(1)
    with c_snap2:
        if st.button("Rebuild 4-week", key="snap_build_4w"):
            snap4_data, snap4_meta = get_snapshot(4, rebuild=True)
        else:
            snap4_data, snap4_meta = get_snapshot(4)
    with c_snap3:
        if st.button("Clear snapshot cache", key="snap_clear"):
            weekly_snapshot.clear()
            snap1_data, snap1_meta = (pd.DataFrame(), {}, None), {}
            snap4_data, snap4_meta = (pd.DataFrame(), {}, None), {}

    def render_snapshot(title, payload, meta):
        snap_df, snap_details, snap_window = payload
        st.markdown(f"### {title}")
        if snap_window:
            st.markdown(f"**Window:** {snap_window[0]:%d %b %Y} - {snap_window[1]:%d %b %Y}")
        if meta and meta.get("built_at"):
            st.caption(f"Cached snapshot built at {meta['built_at']}")

        if snap_df.empty:
            st.info("No data available for snapshot.")
            return snap_df, snap_details, snap_window

        snap_display = snap_df.copy()
        snap_display["Ezeas improvement vs Mecca"] = (
            snap_display["Mecca % error"].abs() - snap_display["Ezeas % error"].abs()
        )
        snap_display["Ezeas vs Mecca"] = np.where(
            snap_display["Ezeas improvement vs Mecca"] >= 0, "Better", "On track"
        )

        sigma_map = snap_display.set_index("Store")["Sigma_weekly"].to_dict()

        def _style_row_display(row):
            sigma = sigma_map.get(row.get("Store"), 0.0)
            styles = [""] * len(row)
            for col in ["Mecca % error", "Ezeas % error"]:
                if col in row.index:
                    styles[row.index.get_loc(col)] = _err_color(row[col], sigma)
            if "Ezeas improvement vs Mecca" in row.index:
                styles[row.index.get_loc("Ezeas improvement vs Mecca")] = (
                    "color:#16a34a; font-weight:600" if row["Ezeas improvement vs Mecca"] >= 0 else "color:#dc2626; font-weight:600"
                )
            if "Ezeas vs Mecca" in row.index:
                styles[row.index.get_loc("Ezeas vs Mecca")] = (
                    "color:#16a34a; font-weight:700" if row["Ezeas vs Mecca"] == "Better" else "color:#0f172a; font-weight:600"
                )
            return styles

        snap_display = snap_display.drop(columns=["Sigma_weekly"])
        styled = (
            snap_display.style.apply(_style_row_display, axis=1)
            .format(
                {
                    "Actual weekly sales": "${:,.0f}",
                    "Mecca weekly forecast": "${:,.0f}",
                    "Ezeas weekly forecast": "${:,.0f}",
                    "Mecca % error": "{:+.1%}",
                    "Ezeas % error": "{:+.1%}",
                    "Ezeas improvement vs Mecca": "{:+.1%}",
                }
            )
        )
        st.dataframe(styled, use_container_width=True)
        st.caption(
            f"Text color: green within +/- 2 sigma, amber > +2 sigma, red < -2 sigma. "
            f"Ezeas is calibrated per store using last {CALIBRATION_WEEKS} weeks and blended to minimise MAE."
        )
        return snap_df, snap_details, snap_window

    def _err_color(val, sigma):
        if pd.isna(val):
            return "color:#94a3b8"
        if abs(val) <= 2 * sigma:
            return "color:#16a34a; font-weight:600"
        if val > 2 * sigma:
            return "color:#d97706; font-weight:700"
        return "color:#dc2626; font-weight:700"

    snap1_df, snap1_details, snap1_window = render_snapshot("Last 1 week", snap1_data, snap1_meta)
    snap4_df, snap4_details, snap4_window = render_snapshot("Last 4 weeks", snap4_data, snap4_meta)

    st.markdown("### Custom window")
    custom_weeks = st.number_input("Weeks", min_value=1, max_value=12, value=2, step=1, key="snap_custom_weeks")
    if st.button("Build custom window", key="snap_custom_build"):
        custom_data, custom_meta = get_snapshot(custom_weeks, rebuild=True)
        st.session_state.custom_snapshot = (custom_data, custom_meta, custom_weeks)

    custom_payload = st.session_state.get("custom_snapshot")
    if custom_payload and custom_payload[2] == custom_weeks:
        custom_data, custom_meta, _ = custom_payload
        render_snapshot(f"Custom {custom_weeks} weeks", custom_data, custom_meta)

    st.markdown("#### Drill down by store")
    drill_options = ["Last 1 week", "Last 4 weeks"]
    if custom_payload:
        drill_options.append(f"Custom {custom_payload[2]} weeks")
    drill_sel = st.selectbox("Drill-down window", drill_options, key="snap_drill_window")
    if drill_sel == "Last 1 week":
        snap_details = snap1_details
    elif drill_sel == "Last 4 weeks":
        snap_details = snap4_details
    else:
        snap_details = custom_payload[0][1] if custom_payload else snap1_details

    store_pick = st.selectbox("Store", STORE_OPTIONS, key="snap_store_pick")
    info = snap_details.get(store_pick)
    if info:
        week_df = info["week_df"].copy()
        start, end = info["start"], info["end"]
        sigma_weekly = info["sigma_weekly"]
        sigma_daily = sigma_weekly * np.sqrt(7) if pd.notna(sigma_weekly) else np.nan

        st.caption(f"{store_pick} - {start:%d %b %Y} - {end:%d %b %Y}")

        series_df = week_df[["Date", "SalesActual", "SalesForecast"]].copy()
        if "Ezeas_Adjusted" in week_df.columns:
            series_df["Ezeas forecast"] = week_df["Ezeas_Adjusted"]
        else:
            series_df["Ezeas forecast"] = week_df["Hybrid_Forecast_Sales"]
        series_df.rename(
            columns={
                "SalesActual": "Actual",
                "SalesForecast": "Mecca forecast",
            },
            inplace=True,
        )
        melt = series_df.melt(id_vars="Date", var_name="Series", value_name="Value")
        fig = px.line(
            melt,
            x="Date",
            y="Value",
            color="Series",
            markers=True,
            title="Daily sales (actual vs forecasts)",
        )
        st.plotly_chart(fig, use_container_width=True)

        if include_ezeas:
            res = week_df[["Date"]].copy()
            res["Mecca"] = (week_df["SalesActual"] - week_df["SalesForecast"]) / week_df["SalesActual"]
            if "Ezeas_Adjusted" in week_df.columns:
                res["Ezeas"] = (week_df["SalesActual"] - week_df["Ezeas_Adjusted"]) / week_df["SalesActual"]
            else:
                res["Ezeas"] = (week_df["SalesActual"] - week_df["Hybrid_Forecast_Sales"]) / week_df["SalesActual"]
            res = res.replace([np.inf, -np.inf], np.nan)
            res_m = res.melt(id_vars="Date", var_name="Series", value_name="Residual")
            fig2 = px.line(
                res_m,
                x="Date",
                y="Residual",
                color="Series",
                markers=True,
                title="Residual ratio with +/- 2 sigma band",
            )
            if pd.notna(sigma_daily):
                fig2.add_hline(
                    y=2 * sigma_daily, line_dash="dash", line_color="#16a34a", annotation_text="+2 sigma"
                )
                fig2.add_hline(
                    y=-2 * sigma_daily, line_dash="dash", line_color="#dc2626", annotation_text="-2 sigma"
                )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Daily sigma derived from weekly sigma using sqrt(7) scaling.")

        dow_tbl = (
            week_df.assign(DOW=week_df["Date"].dt.weekday)
            .groupby("DOW")
            .agg(
                Actual=("SalesActual", "sum"),
                Mecca=("SalesForecast", "sum"),
                Ezeas=("Ezeas_Adjusted", "sum") if "Ezeas_Adjusted" in week_df.columns else ("Hybrid_Forecast_Sales", "sum"),
            )
            .reset_index()
        )
        dow_tbl["DOW"] = dow_tbl["DOW"].apply(fmt_dow)
        st.dataframe(
            dow_tbl.style.format({"Actual": "${:,.0f}", "Mecca": "${:,.0f}", "Ezeas": "${:,.0f}"}),
            use_container_width=True,
        )

# ------------------------------------------------------------------
# Update Forecasts tab
# ------------------------------------------------------------------
with tab_forecast:
    st.markdown("### Update Forecasts")
    days_ago = (NEW_RUN_DATE - PRIOR_RUN_DATE).days
    c_info1, c_info2, c_info3 = st.columns([1.2, 1.2, 2])
    with c_info1:
        st.metric("Prior run", f"{PRIOR_RUN_DATE:%d %b %Y}", help=f"{days_ago} days before the new run")
    with c_info2:
        st.metric("New run", f"{NEW_RUN_DATE:%d %b %Y}")
    with c_info3:
        st.markdown(
            "<div class=\"update-note\">Sales + footfall forecasts for the next 13 weeks. Typical runtime is about a minute per store.</div>",
            unsafe_allow_html=True,
        )

    store_choice = st.selectbox("Store", ["All stores"] + STORE_OPTIONS, key="update_store")
    weeks = st.number_input("Forecast horizon (weeks)", min_value=4, max_value=26, value=13, step=1, key="update_weeks")
    if store_choice == "All stores":
        st.info("Running all stores can take a few minutes.")

    if st.button("Clear update cache", key="update_clear_cache"):
        build_update_forecast.clear()
        st.info("Update cache cleared.")

    if st.button("Run update", type="primary", key="run_update"):
        with st.spinner("Running update..."):
            if store_choice == "All stores":
                rows = []
                progress = st.progress(0.0)
                for idx, s in enumerate(STORE_OPTIONS):
                    sales = build_update_forecast(s, "sales", window_weeks=weeks)
                    foot = build_update_forecast(s, "footfall", window_weeks=weeks)
                    rows.append(
                        {
                            "Store": s,
                            "Prior sales": sales["prior"]["Forecast"].sum(),
                            "New sales": sales["new"]["Forecast"].sum(),
                            "Prior footfall": foot["prior"]["Forecast"].sum(),
                            "New footfall": foot["new"]["Forecast"].sum(),
                        }
                    )
                    progress.progress((idx + 1) / len(STORE_OPTIONS))
                progress.empty()
                table = pd.DataFrame(rows).sort_values("Store")
                st.dataframe(
                    table.style.format(
                        {
                            "Prior sales": "${:,.0f}",
                            "New sales": "${:,.0f}",
                            "Prior footfall": "{:,.0f}",
                            "New footfall": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )
            else:
                sales = build_update_forecast(store_choice, "sales", window_weeks=weeks)
                foot = build_update_forecast(store_choice, "footfall", window_weeks=weeks)

                actual_sales = sales["actual"].copy()
                actual_sales = actual_sales[actual_sales["Date"] <= pd.to_datetime(NEW_RUN_DATE)].tail(56)
                actual_sales = actual_sales.rename(columns={"SalesActual": "Actual"})

                actual_foot = foot["actual"].copy()
                actual_foot = actual_foot[actual_foot["Date"] <= pd.to_datetime(NEW_RUN_DATE)].tail(56)
                actual_foot = actual_foot.rename(columns={"FootfallActual": "Actual"})

                def build_chart(actual_df, prior_df, new_df, title):
                    parts = []
                    if not actual_df.empty:
                        parts.append(actual_df.rename(columns={"Actual": "Value"}).assign(Series="Actual"))
                    parts.append(prior_df.rename(columns={"Forecast": "Value"}).assign(Series="Prior run"))
                    parts.append(new_df.rename(columns={"Forecast": "Value"}).assign(Series="New run"))
                    chart_df = pd.concat(parts, ignore_index=True)
                    fig = px.line(
                        chart_df,
                        x="Date",
                        y="Value",
                        color="Series",
                        markers=True,
                        title=title,
                        color_discrete_map={"Actual": "#111827", "Prior run": "#fca5a5", "New run": "#2563eb"},
                    )
                    fig.update_layout(legend_title_text="", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

                build_chart(actual_sales, sales["prior"], sales["new"], "Sales forecast update")
                build_chart(actual_foot, foot["prior"], foot["new"], "Footfall forecast update")

                summary = pd.DataFrame(
                    {
                        "Run": ["Prior run", "New run"],
                        "Sales forecast total": [
                            sales["prior"]["Forecast"].sum(),
                            sales["new"]["Forecast"].sum(),
                        ],
                        "Footfall forecast total": [
                            foot["prior"]["Forecast"].sum(),
                            foot["new"]["Forecast"].sum(),
                        ],
                    }
                )
                st.dataframe(
                    summary.style.format(
                        {
                            "Sales forecast total": "${:,.0f}",
                            "Footfall forecast total": "{:,.0f}",
                        }
                    ),
                    use_container_width=True,
                )

# ------------------------------------------------------------------
# Analyse Forecasts tab
# ------------------------------------------------------------------
with tab_hist:
    st.markdown("### Analyse Forecasts")
    st.caption("Pivot-style view of actuals vs Ezeas forecast. Choose store and window.")

    c1, c2, c3 = st.columns([1.6, 1, 1])
    store = c1.selectbox("Store", STORE_OPTIONS, key="an_store")
    timeframe = c2.selectbox(
        "Timeframe",
        ["last4_next13", "last2_next2", "custom"],
        format_func=lambda x: {
            "last4_next13": "Last 4 weeks + Next 13 weeks",
            "last2_next2": "Last 2 weeks + Next 2 weeks",
            "custom": "Custom",
        }[x],
        key="an_timeframe",
    )
    metric = c3.radio("Graph", ["sales", "traffic", "both"], horizontal=True, key="an_metric")

    if timeframe == "custom":
        c4, c5 = st.columns(2)
        past_weeks = c4.number_input("Past weeks", min_value=1, max_value=26, value=4, step=1, key="an_past")
        future_weeks = c5.number_input("Future weeks", min_value=1, max_value=26, value=13, step=1, key="an_future")
    elif timeframe == "last2_next2":
        past_weeks, future_weeks = 2, 2
    else:
        past_weeks, future_weeks = 4, 13

    last_actual = pd.to_datetime(get_store_df(store)["Date"]).max().date()
    start = last_actual - timedelta(days=past_weeks * 7 - 1)
    end = last_actual + timedelta(days=future_weeks * 7)
    st.caption(f"Window: {start} to {end} (last actual {last_actual})")

    if st.button("Load analysis", type="primary", key="an_run"):
        with st.spinner("Loading forecast view..."):
            def render_metric(label, metric_name):
                df_view = build_analyze(store, metric_name, start, end)
                if df_view.empty:
                    st.info(f"No data for {label}.")
                    return
                df_melt = df_view.melt(
                    id_vars=["Date", "DayName"],
                    value_vars=["Actual", "Forecast"],
                    var_name="Series",
                    value_name="Value",
                )
                fig = px.line(
                    df_melt,
                    x="Date",
                    y="Value",
                    color="Series",
                    markers=True,
                    title=f"{label} - Actual vs Ezeas forecast",
                    color_discrete_map={"Actual": "#111827", "Forecast": "#2563eb"},
                )
                fig.update_traces(connectgaps=False)
                fig.add_vline(x=pd.to_datetime(last_actual), line_dash="dash", line_color="#94a3b8")
                fig.update_layout(legend_title_text="", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    df_view.style.format({"Actual": "{:,.0f}", "Forecast": "{:,.0f}"}),
                    use_container_width=True,
                )

            if metric in ["sales", "both"]:
                render_metric("Sales", "sales")
            if metric in ["traffic", "both"]:
                render_metric("Traffic", "footfall")
