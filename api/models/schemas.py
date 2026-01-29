from datetime import date
from typing import List, Optional
from pydantic import BaseModel, Field

class ForecastRunRequest(BaseModel):
    store: str
    metric: str = Field("sales", pattern="^(sales|footfall)$")
    horizon: str = Field("next_4w")  # last_next_2w | next_4w | next_13w | custom
    start: Optional[date] = None
    end: Optional[date] = None
    dow_filter: Optional[List[int]] = None  # 0=Mon .. 6=Sun
    quantum_hours: float = 6.0
    blend_weight: float = 0.3
    include_full: bool = False
    download: Optional[str] = Field(None, pattern="^(csv|jsonl)$")

class ForecastSeries(BaseModel):
    date: date
    series: str
    value: float

class ForecastRunResponse(BaseModel):
    kpis: dict
    sample: list
    series: List[ForecastSeries]
    metadata: dict
    full: Optional[list] = None
    csv: Optional[str] = None

class ResidualResponse(BaseModel):
    metrics: dict
    histogram: dict
    weekly: list
    dow_series: list

class GridCell(BaseModel):
    store: str
    dow: int
    residual: float
    sigma: float
    level: str  # green|amber|red

class GridResponse(BaseModel):
    week_start: date
    week_end: date
    cells: List[GridCell]

class StorePerformanceResponse(BaseModel):
    metrics: dict
    series: list
