import pandas as pd
from .quantum_forecast import QuantumForecast

def run_qf(df: pd.DataFrame, metric: str, start, end, quantum_hours: float, blend_weight: float):
    work = df.copy()
    if metric == "footfall":
        work["SalesActual"] = work["FootfallActual"]
        work["SalesForecast"] = work["SalesForecast"].fillna(
            work["SalesActual"].rolling(7, min_periods=1).mean()
        )
        # Provide a minimal positive proxy for labour to avoid zero-slope issues
        proxy_hours = (work["SalesActual"] * 0.02).clip(lower=1)
        work["HoursActual"] = proxy_hours
        work["HoursForecast"] = proxy_hours
    qf = QuantumForecast(
        intput_df=work[["Date","SalesActual","SalesForecast","HoursActual","HoursForecast"]],
        forecast_start=start,
        forecast_end=end,
        quantum_hours=quantum_hours,
        blend_weight=blend_weight,
        name=str(work["store_name"].iloc[0]) if "store_name" in work else "Store",
    )
    qf.run()
    output = qf.output_data

    # Post-process: zero-out forecasts on key AU public holidays
    def is_au_public_holiday(dt: pd.Timestamp) -> bool:
        if dt.month == 1 and dt.day == 1:  # New Year
            return True
        if dt.month == 1 and dt.day == 26:  # Australia Day (fixed date)
            return True
        if dt.month == 4 and dt.day == 25:  # ANZAC Day
            return True
        if dt.month == 12 and dt.day == 25:  # Christmas
            return True
        # Movables: Good Friday, Easter Sunday, Easter Monday
        year = dt.year
        easter = pd.Timestamp(year, 3, 21) + pd.offsets.Easter()  # Easter Sunday
        good_friday = easter - pd.Timedelta(days=2)
        easter_mon = easter + pd.Timedelta(days=1)
        return dt.normalize() in {good_friday.normalize(), easter.normalize(), easter_mon.normalize()}

    data = output.get("Data", [])
    for row in data:
        try:
            dt = pd.to_datetime(row.get("Date"))
            if is_au_public_holiday(dt):
                for key in [
                    "Hybrid_Forecast_Sales",
                    "RandomForest_Forecast",
                    "Intuitive_Forecast_Sales",
                    "Hybrid_Forecast_Labour",
                    "Hybrid_Labour",
                ]:
                    if key in row:
                        row[key] = 0
        except Exception:
            continue
    output["Data"] = data
    return output
