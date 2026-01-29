import pandas as pd
from functools import lru_cache
from pathlib import Path
from ..core.config import DATA_FILE

@lru_cache(maxsize=1)
def load_all(mtime: float = None) -> pd.DataFrame:
    df = pd.read_excel(DATA_FILE, sheet_name="Data")
    df["date"] = pd.to_datetime(df["date"])
    return df

def list_stores() -> list[str]:
    df = load_all(Path(DATA_FILE).stat().st_mtime)
    return sorted(df["store_name"].unique())

def get_store_df(store: str) -> pd.DataFrame:
    df = load_all(Path(DATA_FILE).stat().st_mtime)
    sub = df[df["store_name"] == store].copy()
    if sub.empty:
        raise ValueError(f"Unknown store {store}")
    sub.rename(
        columns={
            "date": "Date",
            "actual_sales": "SalesActual",
            "target_sales": "SalesForecast",
            "productive_hours": "HoursActual",
            "target_productive_hours": "HoursForecast",
            "Footfall": "FootfallActual",
        },
        inplace=True,
    )
    sub["Date"] = pd.to_datetime(sub["Date"])
    return sub
