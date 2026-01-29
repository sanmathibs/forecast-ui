from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
from ..models.schemas import ResidualResponse
from ..services.data_loader import get_store_df
from ..services.metrics import residual_stats
from ..services.forecast_engine import run_qf

router = APIRouter()

@router.get("/residuals", response_model=ResidualResponse)
def residuals(store: str, dow: int | None = None):
    try:
        df = get_store_df(store)
    except ValueError as e:
        raise HTTPException(404, str(e))
    cutoff = df["Date"].max() - pd.Timedelta(weeks=53)
    df = df[df["Date"] >= cutoff]

    start = df["Date"].min().date()
    end = df["Date"].max().date()
    ours = run_qf(df, "sales", start, end, 6.0, 0.3)
    fdf = pd.DataFrame(ours["Data"])
    fdf["Date"] = pd.to_datetime(fdf["Date"])
    merged = df.merge(fdf[["Date","Hybrid_Forecast_Sales"]], on="Date", how="left")
    if dow is not None:
        merged = merged[merged["Date"].dt.weekday == dow]

    stats_their, resid_their = residual_stats(merged["SalesActual"], merged["SalesForecast"])
    stats_ours, resid_ours = residual_stats(merged["SalesActual"], merged["Hybrid_Forecast_Sales"])

    hist_their, bins_their = np.histogram(resid_their, bins=20) if len(resid_their) else ([], [])
    hist_ours, bins_ours = np.histogram(resid_ours, bins=20) if len(resid_ours) else ([], [])

    weekly = (
        merged.assign(Week=lambda x: x["Date"] - pd.to_timedelta(x["Date"].dt.weekday, unit="d"))
        .groupby("Week")
        .agg(
            Actual=("SalesActual","sum"),
            TheirForecast=("SalesForecast","sum"),
            OurForecast=("Hybrid_Forecast_Sales","sum")
        )
        .reset_index()
        .to_dict(orient="records")
    )
    dow_series = (
        merged.assign(DOW=lambda x: x["Date"].dt.weekday)
        .groupby("DOW")
        .agg(mean_residual=("SalesForecast", lambda s: float((merged["SalesActual"]-s).mean())))
        .reset_index()
        .to_dict(orient="records")
    )
    return ResidualResponse(
        metrics={"their": stats_their, "ours": stats_ours},
        histogram={"their": hist_their.tolist() if len(hist_their) else [], "ours": hist_ours.tolist() if len(hist_ours) else []},
        weekly=weekly,
        dow_series=dow_series,
    )
