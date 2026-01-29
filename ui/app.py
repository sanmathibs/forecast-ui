import os
import sys
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


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def require_login():
    if "authed" not in st.session_state:
        st.session_state.authed = False
    if st.session_state.authed:
        return True
    st.markdown(
        """
        <style>
        .login-card {
            background: linear-gradient(140deg, #e0f2fe, #eef2ff);
            color: #0f172a;
            padding: 28px;
            border-radius: 16px;
            box-shadow: 0 16px 30px rgba(15,23,42,0.12);
            max-width: 420px;
            margin: 40px auto;
            border: 1px solid #dbeafe;
        }
        .login-card h2 { margin: 0 0 8px 0; font-size: 24px; }
        .login-card p { margin: 0 0 18px 0; opacity: 0.8; }
        .login-card input {
            background: #fff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.markdown(
            """
            <div class="login-card">
              <h2>Welcome back</h2>
              <p>Sign in to explore forecasts, prove accuracy, and see performance at a glance.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    c_l1, c_l2 = st.columns(2)
    user = c_l1.text_input("Username", value=APP_DEFAULT_USER, key="login_user",
                           help="Enter your username", placeholder="your.name")
    pw = c_l2.text_input("Password", type="password", key="login_pw", placeholder="••••••••")
    if st.button("Enter", type="primary", key="login_btn"):
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
    .hero-chip {
        padding: 8px 14px;
        border-radius: 999px;
        background: #2563eb;
        color: #f8fafc;
        font-size: 13px;
        border: 1px solid #1d4ed8;
    }
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
    </style>
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-left">
        <h1>{APP_TITLE}</h1>
        <p>See what’s coming, prove we’re sharper than legacy forecasts, and spot store/day drift fast.</p>
      </div>
      <div class="hero-chip">Powered by QuantumForecast · Hybrid ML + Intuitive</div>
    </div>
    """,
    unsafe_allow_html=True,
)

profile = data_profile()
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Stores covered", profile["stores"])
mc2.metric("Latest data", f"{profile['date_max']:%d %b %Y}")
mc3.metric("Avg daily sales", f"${profile.get('avg_sales',0):,.0f}")
mc4.metric("Avg daily footfall", f"{profile.get('avg_foot',0):,.0f}")
if "username" in st.session_state and st.session_state.get("authed"):
    st.markdown(f"**Hi, {st.session_state['username']}!**")

# ------------------------------------------------------------------
# Auth gate
# ------------------------------------------------------------------
if not require_login():
    st.stop()

# ------------------------------------------------------------------
# Tabs with helper tooltips
# ------------------------------------------------------------------
tab_forecast, tab_hist, tab_perf = st.tabs(
    [
        "🔮 Forecast",
        "📜 Historical Analysis",
        "💡 Performance",
    ]
)

# ------------------------------------------------------------------
# Forecast tab
# ------------------------------------------------------------------
with tab_forecast:
    st.caption("Configure a window, run both sales and footfall models, and compare vs Mecca instantly.")
    st.info("Use quick presets or customise below.")
    qc1, qc2 = st.columns(2)
    with qc1:
        if st.button("⚡ Quick: Next 4 weeks (sales)", use_container_width=True, key="quick_sales"):
            st.session_state.quick_preset = {"metric": "sales", "horizon": "next_4w"}
    with qc2:
        if st.button("⚡ Quick: Next 4 weeks (footfall)", use_container_width=True, key="quick_foot"):
            st.session_state.quick_preset = {"metric": "footfall", "horizon": "next_4w"}

    col1, col2, col3, col4 = st.columns([1.4, 1, 1, 1])
    store = col1.selectbox("Store", STORE_OPTIONS, index=STORE_OPTIONS.index(DEFAULT_STORE))
    preset = st.session_state.get("quick_preset")
    metric_default = preset["metric"] if preset else "sales"
    horizon_default = preset["horizon"] if preset else "next_4w"
    metric = col2.radio("Metric", ["sales", "footfall"], horizontal=True, index=0 if metric_default=="sales" else 1)
    horizon = col3.selectbox(
        "Time frame",
        ["last_next_2w", "next_4w", "next_13w", "custom"],
        format_func=lambda x: {
            "last_next_2w": "Last 2w + Next 2w",
            "next_4w": "Next 4 weeks",
            "next_13w": "Next 13 weeks",
            "custom": "Custom",
        }[x],
        index=["last_next_2w","next_4w","next_13w","custom"].index(horizon_default),
    )
    dow_filter = dow_multiselect("forecast_dow")
    start = end = None
    if horizon == "custom":
        c1, c2 = st.columns(2)
        start = c1.date_input("Start date", value=datetime.today().date())
        end = c2.date_input(
            "End date", value=datetime.today().date() + timedelta(days=28)
        )
    quantum_hours = col4.slider("Quantum hours", 3.0, 8.0, 6.0, 0.5)
    blend_weight = st.slider("Blend (0=RF,1=Intuitive)", 0.0, 1.0, 0.3, 0.05)

    run_clicked = st.button("Run forecast", type="primary")
    if run_clicked:
        if not start or not end:
            # derive from preset
            last_actual = pd.to_datetime(get_store_df(store)["Date"]).max().date()
            if horizon == "last_next_2w":
                start = last_actual + timedelta(days=1)
                end = start + timedelta(days=13)
            elif horizon == "next_4w":
                start = last_actual + timedelta(days=1)
                end = start + timedelta(days=27)
            elif horizon == "next_13w":
                start = last_actual + timedelta(days=1)
                end = start + timedelta(days=90)
        try:
            with st.spinner("Running forecast (sales + footfall)...may take up to 2 minutes"):
                with st.status("Rolling models...", expanded=False) if hasattr(st, "status") else st.spinner("Rolling models..."):
                    # Always run both sales and footfall for comparison
                    out_sales, full_sales, future_sales = run_forecast(
                        store,
                        "sales",
                        start,
                        end,
                        quantum_hours,
                        blend_weight,
                        dow_filter=dow_filter,
                    )
                    out_foot, full_foot, future_foot = run_forecast(
                        store,
                        "footfall",
                        start,
                        end,
                        quantum_hours,
                        blend_weight,
                        dow_filter=dow_filter,
                    )
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()
        # Choose which to display as primary (match user selection)
        if metric == "footfall":
            full, future, out = full_foot, future_foot, out_foot
        else:
            full, future, out = full_sales, future_sales, out_sales

        st.subheader("KPIs")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Days", len(future))
        k2.metric("Total forecast", f"${future['Hybrid_Forecast_Sales'].sum():,.0f}")
        k3.metric("Avg per day", f"${future['Hybrid_Forecast_Sales'].mean():,.0f}")
        if not future.empty:
            peak_idx = future["Hybrid_Forecast_Sales"].idxmax()
            peak_date = future.loc[peak_idx, "Date"].date()
            k4.metric("Peak day", f"{peak_date}")
        else:
            k4.metric("Peak day", "—")
        # Quick description line
        if not future.empty:
            st.caption(
                f"Blend {blend_weight:.0%} intuitive / {1-blend_weight:.0%} RF · Quantum {quantum_hours}h · Window {start} → {end}"
            )

        st.subheader("Sample (first 21 rows)")
        start_dt = pd.to_datetime(start)
        tail_slice = full[
            (full["Date"] >= start_dt - pd.Timedelta(days=14))
            & (full["Date"] < start_dt)
        ][["Date", "DayName", "SalesActual", "SalesForecast", "Hybrid_Forecast_Sales"]].copy()
        tail_slice["Type"] = "Actual (last 14 days)"

        forecast_slice = future[
            ["Date", "DayName", "Hybrid_Forecast_Sales", "SalesForecast"]
        ].copy()
        forecast_slice["Type"] = "Forecast (next window)"

        combined = pd.concat([tail_slice, forecast_slice], ignore_index=True)
        combined = combined.rename(
            columns={
                "SalesActual": "Actual",
                "Hybrid_Forecast_Sales": "Our forecast",
                "SalesForecast": "Mecca forecast",
            }
        )
        st.dataframe(combined.head(30))

        st.subheader("Chart")
        if not future.empty:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            window_start = start_dt - pd.Timedelta(days=14)

            # Slice actuals (last 14 days) and forecasts (next horizon)
            hist_slice = full[(full["Date"] >= window_start) & (full["Date"] < start_dt)][
                ["Date", "SalesActual", "SalesForecast", "Hybrid_Forecast_Sales"]
            ].copy()
            fc_slice = full[(full["Date"] >= start_dt) & (full["Date"] <= end_dt)][
                ["Date", "Hybrid_Forecast_Sales", "SalesForecast"]
            ].copy()

            # Prepare tidy dataframe
            parts = []
            if not hist_slice.empty:
                parts.append(
                    hist_slice.rename(columns={"SalesActual": "Value"}).assign(Series="Actual")
                )
                parts.append(
                    hist_slice.rename(columns={"Hybrid_Forecast_Sales": "Value"}).assign(Series="Our forecast")
                )
                parts.append(
                    hist_slice.rename(columns={"SalesForecast": "Value"}).assign(Series="Mecca forecast")
                )
            if not fc_slice.empty:
                parts.append(
                    fc_slice.rename(columns={"Hybrid_Forecast_Sales": "Value"}).assign(Series="Our forecast")
                )
                parts.append(
                    fc_slice.rename(columns={"SalesForecast": "Value"}).assign(Series="Mecca forecast")
                )

            chart_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["Date","Series","Value"])
            chart_df["DoW"] = chart_df["Date"].dt.day_name()
            fig = px.line(
                chart_df,
                x="Date",
                y="Value",
                color="Series",
                markers=True,
                hover_data={"DoW": True},
                color_discrete_map={
                    "Actual": "#111827",
                    "Our forecast": "#2563eb",
                    "Mecca forecast": "#a855f7",
                },
            )
            # highlight comparison window
            fig.add_vrect(x0=window_start, x1=end_dt, fillcolor="#e0f2fe", opacity=0.15, layer="below", line_width=0)
            fig.add_vline(x=start_dt, line_dash="dash", line_color="#6b7280")
            fig.update_layout(
                legend_title_text="",
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Cross-metric comparison: show both sales and footfall forecasts for the window.
        if not future_sales.empty and not future_foot.empty:
            st.subheader("Sales vs Footfall (forecast window)")
            comp = pd.DataFrame({"Date": future_sales["Date"].values})
            comp["Sales_Forecast"] = future_sales["Hybrid_Forecast_Sales"].values
            comp["Footfall_Forecast"] = future_foot["Hybrid_Forecast_Sales"].values
            # Scale footfall to sales magnitude for visual correlation
            if comp["Footfall_Forecast"].max() > 0:
                scale = comp["Sales_Forecast"].mean() / comp["Footfall_Forecast"].mean()
                comp["Footfall_Forecast_Scaled"] = comp["Footfall_Forecast"] * scale
            else:
                comp["Footfall_Forecast_Scaled"] = comp["Footfall_Forecast"]
            comp_melt = comp.melt(id_vars="Date", var_name="Series", value_name="Value")
            comp_melt["Series"] = comp_melt["Series"].replace(
                {"Footfall_Forecast_Scaled": "Footfall (scaled)", "Footfall_Forecast": "Footfall"}
            )
            fig_fs = px.line(
                comp_melt,
                x="Date",
                y="Value",
                color="Series",
                markers=True,
                color_discrete_map={
                    "Sales_Forecast": "#2563eb",
                    "Footfall_Forecast": "#f59e0b",
                    "Footfall (scaled)": "#f59e0b",
                },
            )
            fig_fs.update_layout(legend_title_text="", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_fs, use_container_width=True)
            # Correlation on unscaled series
            corr = comp["Sales_Forecast"].corr(comp["Footfall_Forecast"])
            st.caption(f"Sales vs Footfall correlation (forecast window): {corr:.2f}")

        st.download_button(
            "Download full window (CSV)",
            data=future.to_csv(index=False),
            file_name=f"{store}_forecast_{start}_to_{end}.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download full window (JSON)",
            data=future.to_json(orient="records", date_format="iso"),
            file_name=f"{store}_forecast_{start}_to_{end}.json",
            mime="application/json",
        )

        # Sales vs Mecca forecast lift on this window (sales metric only)
        if metric == "sales" and not future_sales.empty:
            total_ours = future_sales["Hybrid_Forecast_Sales"].sum()
            total_mecca = future_sales["SalesForecast"].sum()
            if total_mecca != 0:
                lift_pct = (total_ours - total_mecca) / total_mecca * 100
                st.info(f"Our forecast is {lift_pct:+.1f}% vs Mecca over this window (total sales).")

    # -----------------------------
    # Backtest: compare on holdout
    # -----------------------------
    st.markdown("---")
    st.subheader("Backtest (holdout window)")
    bt_weeks = st.number_input("Holdout length (weeks)", min_value=1, max_value=52, value=2, step=1,
                               help="Forecast the last N weeks and compare vs actuals.")
    if st.button("Run backtest", type="primary", key="backtest"):
        df_bt = get_store_df(store)
        last_actual = pd.to_datetime(df_bt["Date"]).max().date()
        holdout_days = bt_weeks * 7
        bt_start = last_actual - timedelta(days=holdout_days) + timedelta(days=1)
        bt_end = last_actual
        try:
            with st.spinner("Running backtest..."):
                out_bt, full_bt, future_bt = run_forecast(
                    store,
                    "sales",
                    bt_start,
                    bt_end,
                    quantum_hours,
                    blend_weight,
                    dow_filter=dow_filter,
                )
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            st.stop()

        # Align actuals for holdout window
        actuals_bt = df_bt[(pd.to_datetime(df_bt["Date"]).dt.date >= bt_start) & (pd.to_datetime(df_bt["Date"]).dt.date <= bt_end)].copy()
        future_bt = future_bt.copy()
        future_bt["Date"] = pd.to_datetime(future_bt["Date"]).dt.date
        actuals_bt["Date"] = pd.to_datetime(actuals_bt["Date"]).dt.date
        merged_bt = actuals_bt.merge(
            future_bt[["Date", "Hybrid_Forecast_Sales", "SalesForecast"]],
            on="Date",
            how="left",
            suffixes=("_act", "_pred"),
        )
        # Construct Mecca forecast: prefer predicted SalesForecast; fallback to actuals' legacy forecast if present
        merged_bt["Mecca forecast"] = merged_bt.get("SalesForecast_pred", pd.Series(index=merged_bt.index))
        if "SalesForecast_act" in merged_bt.columns:
            merged_bt["Mecca forecast"] = merged_bt["Mecca forecast"].fillna(merged_bt["SalesForecast_act"])
        merged_bt = merged_bt.rename(
            columns={
                "SalesActual": "Actual",
                "Hybrid_Forecast_Sales": "Our forecast",
            }
        )

        # Metrics
        def mstats(actual, pred):
            err = actual - pred
            return {
                "MAE": np.abs(err).mean(),
                "RMSE": np.sqrt((err ** 2).mean()),
            }

        mecca_series = merged_bt["Mecca forecast"] if "Mecca forecast" in merged_bt.columns else pd.Series(0, index=merged_bt.index)
        our_series = merged_bt["Our forecast"] if "Our forecast" in merged_bt.columns else pd.Series(0, index=merged_bt.index)

        m_mecca = mstats(merged_bt["Actual"], mecca_series.fillna(0))
        m_ours = mstats(merged_bt["Actual"], our_series.fillna(0))
        cbt1, cbt2 = st.columns(2)
        cbt1.metric("MAE improvement vs Mecca", f"{(m_mecca['MAE'] - m_ours['MAE']):,.0f}")
        cbt2.metric("RMSE improvement vs Mecca", f"{(m_mecca['RMSE'] - m_ours['RMSE']):,.0f}")

        # Chart (only columns that exist)
        value_cols = [c for c in ["Actual", "Our forecast", "Mecca forecast"] if c in merged_bt.columns]
        chart_bt = merged_bt.melt(id_vars="Date", value_vars=value_cols, var_name="Series", value_name="Value")
        fig_bt = px.line(chart_bt, x="Date", y="Value", color="Series", markers=True,
                         color_discrete_map={"Actual": "#111827", "Our forecast": "#2563eb", "Mecca forecast": "#a855f7"})
        fig_bt.update_layout(legend_title_text="", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bt, use_container_width=True)

        st.dataframe(merged_bt)

# ------------------------------------------------------------------
# Historical Analysis tab
# ------------------------------------------------------------------
with tab_hist:
    st.caption("Last 53 weeks; compares Mecca forecast vs ours. Residual = Actual - Forecast.")
    c1, c2 = st.columns([1.4, 1])
    store = c1.selectbox("Store", STORE_OPTIONS, key="hist_store")
    dow = c2.selectbox(
        "Day-of-week filter", [None, 0, 1, 2, 3, 4, 5, 6], format_func=lambda x: "All" if x is None else fmt_dow(x)
    )
    if st.button("Run historical", type="primary"):
        df = get_store_df(store)
        cutoff = pd.to_datetime(df["Date"]).max() - pd.Timedelta(weeks=53)
        df = df[df["Date"] >= cutoff]
        start = df["Date"].min().date()
        end = df["Date"].max().date()
        out, full, future = run_forecast(
            store, "sales", start, end, quantum_hours=6.0, blend_weight=0.3, dow_filter=[dow] if dow is not None else None
        )
        merged = df.merge(
            full[["Date", "Hybrid_Forecast_Sales"]],
            on="Date",
            how="left",
        )
        if dow is not None:
            merged = merged[merged["Date"].dt.weekday == dow]
        merged["Residual_mecca"] = merged["SalesActual"] - merged["SalesForecast"]
        merged["Residual_ours"] = merged["SalesActual"] - merged["Hybrid_Forecast_Sales"]

        st.subheader("Metrics")
        def stats(res):
            return {
                "MAE": np.abs(res).mean(),
                "RMSE": np.sqrt((res**2).mean()),
                "Mean": res.mean(),
                "Std": res.std(ddof=1),
            }
        mecca_stats = stats(merged["Residual_mecca"])
        our_stats = stats(merged["Residual_ours"])

        c_imp1, c_imp2 = st.columns(2)
        c_imp1.metric("MAE improvement vs Mecca", f"{(mecca_stats['MAE'] - our_stats['MAE']):,.0f}",
                      help="Positive value means our MAE is lower than Mecca.")
        c_imp2.metric("RMSE improvement vs Mecca", f"{(mecca_stats['RMSE'] - our_stats['RMSE']):,.0f}",
                      help="Positive value means our RMSE is lower than Mecca.")

        # Win count
        wins = (np.abs(merged["Residual_ours"]) < np.abs(merged["Residual_mecca"])).sum()
        st.caption(f"Our forecast beats Mecca on {wins} of {len(merged)} days in the 53-week window.")

        # Residual distributions
        st.subheader("Residual distributions")
        resid_melt = pd.DataFrame(
            {
                "Residual": pd.concat([merged["Residual_mecca"], merged["Residual_ours"]], ignore_index=True),
                "Series": ["Mecca"] * len(merged) + ["Ours"] * len(merged),
            }
        )
        hist_fig = px.histogram(resid_melt, x="Residual", color="Series", nbins=30, barmode="overlay",
                                color_discrete_map={"Mecca": "#a855f7", "Ours": "#2563eb"})
        hist_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(hist_fig, use_container_width=True)

        st.json({"Mecca forecast": mecca_stats, "Our forecast": our_stats})

        st.subheader("Weekly comparison (sums)")
        weekly = (
            merged.assign(Week=lambda x: x["Date"] - pd.to_timedelta(x["Date"].dt.weekday, unit="d"))
            .groupby("Week")
            .agg(
                Actual=("SalesActual", "sum"),
                TheirForecast=("SalesForecast", "sum"),
                OurForecast=("Hybrid_Forecast_Sales", "sum"),
            )
            .reset_index()
        )
        st.dataframe(weekly)

        st.subheader("DOW residuals")
        dow_df = (
            merged.assign(DOW=lambda x: x["Date"].dt.weekday)
            .groupby("DOW")
            .agg(
                Residual_mecca=("Residual_mecca", "mean"),
                Residual_ours=("Residual_ours", "mean"),
            )
            .reset_index()
        )
        dow_df["DOW"] = dow_df["DOW"].apply(fmt_dow)
        st.caption("Residual = Actual - Forecast. Negative → over-forecast; Positive → under-forecast.")
        st.dataframe(
            dow_df.style.format({"Residual_mecca": "{:.0f}", "Residual_ours": "{:.0f}"})
            .background_gradient(
                cmap="RdYlGn_r",
                subset=["Residual_mecca", "Residual_ours"],
            )
        )

# ------------------------------------------------------------------
# Performance tab
# ------------------------------------------------------------------
with tab_perf:
    st.markdown("### All stores (last full week)")
    st.caption("Residual = Actual - Mecca forecast. Positive → under-forecast; Negative → over-forecast.")
    if st.button("Load grid", type="primary"):
        df = pd.concat([get_store_df(s).assign(store=s) for s in STORE_OPTIONS])
        end = pd.to_datetime(df["Date"]).max().normalize()
        start = end - pd.Timedelta(days=6)
        week = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
        week["dow"] = week["Date"].dt.weekday
        # sigma from prior 8 weeks
        hist = df[(df["Date"] < start) & (df["Date"] >= start - pd.Timedelta(weeks=8))].copy()
        hist["dow"] = hist["Date"].dt.weekday
        sigmas = {}
        for (store, dow), sub in hist.groupby(["store", "dow"]):
            res = sub["SalesActual"] - sub["SalesForecast"]
            sigmas[(store, dow)] = res.std(ddof=1) if len(res) > 1 else 1e-6
        cells = []
        for (store, dow), sub in week.groupby(["store", "dow"]):
            res = sub["SalesActual"] - sub["SalesForecast"]
            val = res.mean() if len(res) else 0.0
            sigma = sigmas.get((store, dow), 1e-6)
            if abs(val) <= 2 * sigma:
                level = "green"
            elif val > 2 * sigma:
                level = "amber"
            else:
                level = "red"
            cells.append({"store": store, "dow": fmt_dow(dow), "residual": val, "level": level, "sigma": sigma})
        grid = pd.DataFrame(cells)
        if not grid.empty:
            # Summary cards
            within = (grid["level"] == "green").mean()
            worst_over = grid.loc[grid["residual"].idxmin()]
            worst_under = grid.loc[grid["residual"].idxmax()]
            cA, cB, cC = st.columns(3)
            cA.metric("% within ±2σ", f"{within*100:,.1f}%")
            cB.metric("Largest over-forecast", f"{worst_over['residual']:,.0f}",
                      help=f"{worst_over['store']} · {worst_over['dow']}")
            cC.metric("Largest under-forecast", f"{worst_under['residual']:,.0f}",
                      help=f"{worst_under['store']} · {worst_under['dow']}")

            pivot = grid.pivot(index="store", columns="dow", values="residual")
            # Color using each cell's sigma from precomputed sigmas dict
            def color_cell(val, row_store, col_dow):
                sigma = sigmas.get((row_store, col_dow), 1e-6)
                if abs(val) <= 2 * sigma:
                    return "background-color:#d1fae5"
                if val > 2 * sigma:
                    return "background-color:#fcd34d"
                return "background-color:#fecdd3"

            styled = pivot.style.apply(
                lambda s: [
                    color_cell(s[c], s.name, c) for c in s.index
                ],
                axis=1,
            ).format("{:.0f}")
            st.dataframe(styled, use_container_width=True)
            st.caption("Green = within ±2σ; Amber = under-forecast > +2σ; Red = over-forecast < -2σ. σ per store/DOW from prior 8 weeks.")
        else:
            st.info("No data for last week.")

    st.markdown("### Single store")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    store_sel = c1.selectbox("Store", STORE_OPTIONS, key="perf_store")
    timeframe = c2.selectbox(
        "Timeframe",
        ["last_week", "last_quarter", "last_year"],
        format_func=lambda x: {"last_week": "Last week", "last_quarter": "Last quarter", "last_year": "Last year"}[x],
    )
    dow_sel = c3.selectbox(
        "Day-of-week filter",
        [None, 0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: "All" if x is None else fmt_dow(x),
        key="perf_dow",
    )
    if st.button("Run store performance", type="primary"):
        df = get_store_df(store_sel)
        end = pd.to_datetime(df["Date"]).max()
        days = {"last_week": 7, "last_quarter": 91, "last_year": 365}[timeframe]
        start = end - pd.Timedelta(days=days - 1)
        sub = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
        if dow_sel is not None:
            sub = sub[sub["Date"].dt.weekday == dow_sel]
        resid = sub["SalesActual"] - sub["SalesForecast"]
        st.json(
            {
                "mean": resid.mean(),
                "stdev": resid.std(ddof=1),
                "mae": np.abs(resid).mean(),
                "rmse": np.sqrt((resid**2).mean()),
            }
        )
        if not sub.empty:
            sub_plot = sub[["Date"]].copy()
            sub_plot["Residual"] = resid.values
            fig = px.line(sub_plot, x="Date", y="Residual", title="Residual time series")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sub_plot)
        else:
            st.info("No data for selection.")
