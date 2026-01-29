from datetime import date, timedelta

def resolve_horizon(h: str, last_actual: date) -> tuple[date, date]:
    if h == "last_next_2w":
        return last_actual - timedelta(days=13), last_actual + timedelta(days=14)
    if h == "next_4w":
        start = last_actual + timedelta(days=1)
        return start, start + timedelta(days=27)
    if h == "next_13w":
        start = last_actual + timedelta(days=1)
        return start, start + timedelta(days=90)
    raise ValueError("custom")
