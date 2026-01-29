from fastapi import APIRouter
import pandas as pd
from ..models.schemas import GridResponse, GridCell, StorePerformanceResponse
from ..services.data_loader import load_all, get_store_df
from ..services.metrics import residual_stats, sigma_level

router = APIRouter()


def _sigma_lookup(df: pd.DataFrame, end):
    """Compute per-store/day sigma from prior 8 weeks residuals vs target forecast."""
    start_hist = end - pd.Timedelta(weeks=8)
    hist = df[(df["date"] >= start_hist) & (df["date"] < end)].copy()
    hist["dow"] = hist["date"].dt.weekday
    sigmas = {}
    for (store, dow), sub in hist.groupby(["store_name", "dow"]):
        stats, resid = residual_stats(sub["actual_sales"], sub["target_sales"])
        sigmas[(store, dow)] = stats["stdev"] if stats["stdev"] else 1e-6
    return sigmas


@router.get("/grid", response_model=GridResponse)
def grid():
    df = load_all()
    end = df["date"].max().normalize()
    start = end - pd.Timedelta(days=6)
    week = df[(df["date"] >= start) & (df["date"] <= end)].copy()
    week["dow"] = week["date"].dt.weekday
    sigmas = _sigma_lookup(df, end)
    cells = []
    for (store, dow), sub in week.groupby(["store_name", "dow"]):
        stats, resid = residual_stats(sub["actual_sales"], sub["target_sales"])
        res_val = float(resid.mean()) if len(resid) else 0.0
        sigma = sigmas.get((store, int(dow)), 1e-6)
        level = sigma_level(res_val, sigma)
        cells.append(GridCell(store=store, dow=int(dow), residual=res_val, sigma=sigma, level=level))
    return GridResponse(week_start=start.date(), week_end=end.date(), cells=cells)


@router.get("/store", response_model=StorePerformanceResponse)
def store_perf(store: str, timeframe: str = "last_quarter", dow: int | None = None):
    df = get_store_df(store)
    end = df["Date"].max()
    windows = {"last_week":7, "last_quarter":91, "last_year":365}
    days = windows.get(timeframe, 91)
    start = end - pd.Timedelta(days=days-1)
    sub = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    if dow is not None:
        sub = sub[sub["Date"].dt.weekday == dow]
    stats, resid = residual_stats(sub["SalesActual"], sub["SalesForecast"])
    series = sub[["Date"]].assign(residual=resid).to_dict(orient="records")
    return StorePerformanceResponse(metrics=stats, series=series)
