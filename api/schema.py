from pydantic import BaseModel
from typing import Optional

class PredictionInput(BaseModel):
    region: str
    vegetable_commodity: str
    date: Optional[str] = None  

    
    temperature_c: Optional[float] = None
    rainfall_mm: Optional[float] = None
    humidity_pct: Optional[float] = None
    crop_yield_impact_score: Optional[float] = None

   
    month: Optional[int] = None
    weekofyear: Optional[int] = None
    quarter: Optional[int] = None
    year: Optional[int] = None