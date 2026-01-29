import numpy as np
import pandas as pd

def residual_stats(actual, forecast):
    resid = actual - forecast
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mape": float(np.mean(np.abs(resid) / np.where(actual==0, np.nan, actual)) * 100),
        "mean": float(np.mean(resid)),
        "stdev": float(np.std(resid, ddof=1)),
    }, resid

def sigma_level(residual, sigma):
    if abs(residual) <= 2*sigma:
        return "green"
    if residual > 2*sigma:
        return "amber"
    return "red"
