from fastapi import FastAPI
from .routers import forecast, historical, performance

app = FastAPI(title="Store Forecast API", version="1.0.0")

app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
app.include_router(historical.router, prefix="/historical", tags=["Historical"])
app.include_router(performance.router, prefix="/performance", tags=["Performance"])


@app.get("/health")
def health():
    return {"status": "ok"}
