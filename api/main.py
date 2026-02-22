import shap
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd



from api.schema import PredictionInput

ROOT = Path(__file__).resolve().parents[1]  # project root
BASELINE_PATH = ROOT / "models" / "climate_baselines.json"

with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    CLIMATE_BASELINES = json.load(f)
app = FastAPI(title="Vegetable Price Prediction API")

# Enable CORS (so React can connect later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "veg_price_model.joblib"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]


@app.get("/")
def home():
    return {"message": "Vegetable Price Prediction API is running 🚀"}


@app.post("/predict")
def predict_price(data: PredictionInput):
    try:
        input_dict = data.model_dump()  # pydantic v2
        input_dict["region"] = str(input_dict["region"]).strip()
        input_dict["vegetable_commodity"] = str(input_dict["vegetable_commodity"]).strip()

        # ✅ If date is provided, derive time fields automatically
        used_baseline = False
        if input_dict.get("date"):
            y, m, w, q = derive_time_parts(input_dict["date"])
            input_dict["year"] = y
            input_dict["month"] = m
            input_dict["weekofyear"] = w
            input_dict["quarter"] = q

        # ✅ Validate we have month/year (either from date or direct input)
        if input_dict.get("month") is None or input_dict.get("year") is None:
            raise ValueError("Provide either 'date' (YYYY-MM-DD) or 'month' and 'year'.")

         # ✅ Auto-fill climate values if missing (Simple mode)
        if (
            input_dict.get("temperature_c") is None
            or input_dict.get("rainfall_mm") is None
            or input_dict.get("humidity_pct") is None
            or input_dict.get("crop_yield_impact_score") is None
        ):
            base = get_climate_baseline(input_dict["region"], int(input_dict["month"]))
            input_dict["temperature_c"] = base["temperature_c"] if input_dict.get("temperature_c") is None else input_dict["temperature_c"]
            input_dict["rainfall_mm"] = base["rainfall_mm"] if input_dict.get("rainfall_mm") is None else input_dict["rainfall_mm"]
            input_dict["humidity_pct"] = base["humidity_pct"] if input_dict.get("humidity_pct") is None else input_dict["humidity_pct"]
            input_dict["crop_yield_impact_score"] = base["crop_yield_impact_score"] if input_dict.get("crop_yield_impact_score") is None else input_dict["crop_yield_impact_score"]
            used_baseline = True

        # ✅ season derived from month (must match training)
        m = int(input_dict["month"])
        input_dict["season"] = "maha" if m in [10, 11, 12, 1, 2, 3] else "yala"

        df = pd.DataFrame([input_dict])
        df = df[feature_columns]
        pred = float(model.predict(df)[0])

        return {
            "predicted_price_lkr_per_kg": round(pred, 2),
            "used_climate_baseline": used_baseline,
            "final_features": {
                "region": input_dict["region"],
                "vegetable_commodity": input_dict["vegetable_commodity"],
                "year": input_dict["year"],
                "month": input_dict["month"],
                "weekofyear": input_dict["weekofyear"],
                "quarter": input_dict["quarter"],
                "temperature_c": input_dict["temperature_c"],
                "rainfall_mm": input_dict["rainfall_mm"],
                "humidity_pct": input_dict["humidity_pct"],
                "crop_yield_impact_score": input_dict["crop_yield_impact_score"],
                "season": input_dict["season"],
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from datetime import datetime
import numpy as np
import pandas as pd
import shap
from fastapi import HTTPException

@app.post("/explain")
def explain_prediction(data: PredictionInput):
    try:
        input_dict = data.dict()

        # -----------------------------
        # 1) Normalize strings
        # -----------------------------
        input_dict["region"] = str(input_dict.get("region", "")).strip()
        input_dict["vegetable_commodity"] = str(input_dict.get("vegetable_commodity", "")).strip()

        # -----------------------------
        # 2) Derive time features from date
        # -----------------------------
        # Expect date as "YYYY-MM-DD" from frontend
        date_str = input_dict.get("date")
        if not date_str:
            raise ValueError("date is required (YYYY-MM-DD).")

        dt = datetime.strptime(date_str, "%Y-%m-%d")

        input_dict["year"] = int(dt.year)
        input_dict["month"] = int(dt.month)
        input_dict["weekofyear"] = int(dt.isocalendar().week)
        input_dict["quarter"] = int((dt.month - 1) // 3 + 1)

        # Season (same as training)
        m = input_dict["month"]
        input_dict["season"] = "maha" if m in [10, 11, 12, 1, 2, 3] else "yala"

        # -----------------------------
        # 3) Apply baseline climate if missing (Simple mode)
        # -----------------------------
        # If user didn't provide any climate value, we fill from baseline
        # Baselines should be loaded into CLIMATE_BASELINES dict in your API
        # Example structure:
        # CLIMATE_BASELINES[region][month] = {
        #   "temperature_c": ..., "rainfall_mm": ..., "humidity_pct": ..., "crop_yield_impact_score": ...
        # }

        # Check missing values
        climate_fields = ["temperature_c", "rainfall_mm", "humidity_pct", "crop_yield_impact_score"]
        missing_any = any(input_dict.get(f) is None for f in climate_fields)

        used_baseline = False
        baseline_used = {}

        if missing_any:
            region = input_dict["region"]
            month = str(input_dict["month"])

            if region not in CLIMATE_BASELINES or month not in CLIMATE_BASELINES[region]:
                raise ValueError(f"No climate baseline found for region={region}, month={month}")

            base = CLIMATE_BASELINES[region][month]

            for f in climate_fields:
                if input_dict.get(f) is None:
                    input_dict[f] = float(base[f])
                    baseline_used[f] = float(base[f])
                    used_baseline = True

        # -----------------------------
        # 4) Prepare dataframe in correct feature order
        # -----------------------------
        df = pd.DataFrame([{
            "region": input_dict["region"],
            "vegetable_commodity": input_dict["vegetable_commodity"],
            "season": input_dict["season"],
            "temperature_c": float(input_dict["temperature_c"]),
            "rainfall_mm": float(input_dict["rainfall_mm"]),
            "humidity_pct": float(input_dict["humidity_pct"]),
            "crop_yield_impact_score": float(input_dict["crop_yield_impact_score"]),
            "month": int(input_dict["month"]),
            "weekofyear": int(input_dict["weekofyear"]),
            "quarter": int(input_dict["quarter"]),
            "year": int(input_dict["year"]),
        }])

        df = df[feature_columns]  # ensure same order as training

        # -----------------------------
        # 5) Predict
        # -----------------------------
        prediction = float(model.predict(df)[0])

        # -----------------------------
        # 6) SHAP for this single row
        # -----------------------------
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df)  # shape (1, n_features)

        contributions = dict(zip(feature_columns, [float(x) for x in shap_values[0]]))

        # Sort top contributions by absolute impact
        top = sorted(
            [{"feature": k, "shap": v} for k, v in contributions.items()],
            key=lambda x: abs(x["shap"]),
            reverse=True
        )[:10]

        return {
            "predicted_price_lkr_per_kg": round(prediction, 2),
            "used_climate_baseline": used_baseline,
            "baseline_values_used": baseline_used,
            "final_features": df.iloc[0].to_dict(),
            "top_feature_contributions": top,
            "feature_contributions": contributions,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

from datetime import datetime
import math

def derive_time_parts(date_str: str):
    """
    Derive year, month, weekofyear, quarter from ISO date string (YYYY-MM-DD).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.year
    month = dt.month
    weekofyear = int(dt.strftime("%U")) + 1  # simple week number (1-53)
    quarter = math.floor((month - 1) / 3) + 1
    return year, month, weekofyear, quarter


def get_climate_baseline(region: str, month: int):
    """
    Returns baseline climate dict for region+month.
    Fallbacks:
      1) exact region + month
      2) region average across all months
      3) global average across all regions/months
    """
    region_key = region.strip()

    # 1) exact match
    if region_key in CLIMATE_BASELINES and str(month) in CLIMATE_BASELINES[region_key]:
        return CLIMATE_BASELINES[region_key][str(month)]

    # 2) region average across months
    if region_key in CLIMATE_BASELINES:
        vals = list(CLIMATE_BASELINES[region_key].values())
        if vals:
            return {
                "temperature_c": sum(v["temperature_c"] for v in vals) / len(vals),
                "rainfall_mm": sum(v["rainfall_mm"] for v in vals) / len(vals),
                "humidity_pct": sum(v["humidity_pct"] for v in vals) / len(vals),
                "crop_yield_impact_score": sum(v["crop_yield_impact_score"] for v in vals) / len(vals),
            }

    # 3) global average
    all_vals = []
    for reg in CLIMATE_BASELINES.values():
        all_vals.extend(list(reg.values()))
    if all_vals:
        return {
            "temperature_c": sum(v["temperature_c"] for v in all_vals) / len(all_vals),
            "rainfall_mm": sum(v["rainfall_mm"] for v in all_vals) / len(all_vals),
            "humidity_pct": sum(v["humidity_pct"] for v in all_vals) / len(all_vals),
            "crop_yield_impact_score": sum(v["crop_yield_impact_score"] for v in all_vals) / len(all_vals),
        }

    # extreme fallback (should never happen)
    return {"temperature_c": 28.0, "rainfall_mm": 100.0, "humidity_pct": 75.0, "crop_yield_impact_score": 1.0}


@app.get("/health")
def health():
    return {"status": "ok"}