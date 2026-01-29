from fastapi import APIRouter, HTTPException, Response
import pandas as pd
import io
import json
from ..models.schemas import ForecastRunRequest, ForecastRunResponse, ForecastSeries
from ..services.data_loader import list_stores, get_store_df
from ..services.forecast_engine import run_qf
from ..utils.time_windows import resolve_horizon

router = APIRouter()

@router.get("/options")
def options():
    return {
        "stores": list_stores(),
        "metrics": ["sales","footfall"],
        "horizons": ["last_next_2w","next_4w","next_13w","custom"],
        "dow": list(range(7))
    }

@router.post("/run", response_model=ForecastRunResponse)
def run_forecast(req: ForecastRunRequest):
    try:
        df = get_store_df(req.store)
    except ValueError as e:
        raise HTTPException(404, str(e))
    last_actual = pd.to_datetime(df["Date"]).max().date()
    # If explicit dates are provided, honour them regardless of horizon preset.
    if req.start and req.end:
        start, end = req.start, req.end
        if start > end:
            raise HTTPException(400, "start date must be on/before end date")
    else:
        if req.horizon == "custom":
            raise HTTPException(400, "start/end required for custom horizon")
        start, end = resolve_horizon(req.horizon, last_actual)
    out = run_qf(df, req.metric, start, end, req.quantum_hours, req.blend_weight)

    full = pd.DataFrame(out["Data"])
    full["Date"] = pd.to_datetime(full["Date"]).dt.date
    if req.dow_filter:
        full = full[full["Date"].apply(lambda d: d.weekday() in req.dow_filter)]

    future = full[(full["Date"] >= start) & (full["Date"] <= end)]
    kpis = {
        "days": len(future),
        "total": float(future["Hybrid_Forecast_Sales"].sum()),
        "avg": float(future["Hybrid_Forecast_Sales"].mean()) if not future.empty else 0,
        "peak_date": str(future.loc[future["Hybrid_Forecast_Sales"].idxmax(), "Date"]) if not future.empty else None,
    }
    series = [
        ForecastSeries(date=r.Date, series="Hybrid", value=float(r.Hybrid_Forecast_Sales))
        for r in future.itertuples()
    ]
    sample_cols = ["Date","DayName","Hybrid_Forecast_Sales","RandomForest_Forecast","Intuitive_Forecast_Sales"]
    sample = future[sample_cols].head(21).to_dict(orient="records") if not future.empty else []
    full_payload = None
    csv_payload = None
    if req.include_full or req.download:
        full_payload = future.to_dict(orient="records")
        buf = io.StringIO()
        future.to_csv(buf, index=False)
        csv_payload = buf.getvalue()

    # Optional direct download formats
    if req.download == "csv":
        return Response(csv_payload or "", media_type="text/csv")
    if req.download == "jsonl":
        jsonl = "\n".join(json.dumps(row, default=str) for row in full_payload or [])
        return Response(jsonl, media_type="application/json")

    return ForecastRunResponse(
        kpis=kpis,
        sample=sample,
        series=series,
        metadata={
            "ForecastStart": out.get("ForecastStart"),
            "ForecastEnd": out.get("ForecastEnd"),
            "SalesBin": out.get("SalesBin"),
            "SmoothSlope": out.get("SmoothSlope"),
        },
        full=full_payload,
        csv=csv_payload,
    )
