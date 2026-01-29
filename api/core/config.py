from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]           # .../Forecast_application/api
DATA_FILE = ROOT.parent / "Sales Labour Footfall for 12 stores.xlsx"
DEFAULT_QUANTUM_HOURS = 6.0
DEFAULT_BLEND_WEIGHT = 0.3
