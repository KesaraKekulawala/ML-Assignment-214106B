from pydantic import BaseModel
from typing import Optional

class PredictionInput(BaseModel):
    region: str
    vegetable_commodity: str
    date: Optional[str] = None  # e.g. "2026-12-30"

    # optional manual override (Advanced mode)
    temperature_c: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    crop_yield_impact_score: Optional[float] = None

    # optional if user doesn’t send date-derived values
    month: Optional[int] = None
    weekofyear: Optional[int] = None
    quarter: Optional[int] = None
    year: Optional[int] = None