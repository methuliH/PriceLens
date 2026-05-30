#Fast api backend for the vehicle predictor 

import os
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Vehicle Price Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/price_model.joblib"
CURRENT_YEAR = 2026

bundle: dict | None = None


@app.on_event("startup")
async def load_model():
    global bundle
    if not os.path.exists(MODEL_PATH):
        log.warning(f"Model file not found at {MODEL_PATH} — run train.py first. API will start without it.")
        return
    try:
        bundle = joblib.load(MODEL_PATH)
        log.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        log.warning(f"Failed to load model: {e}")


# ── Pydantic models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    make: str
    year: int = Field(..., ge=1980, le=2026)
    mileage: float = Field(..., ge=0)
    fuel_type: str = "Petrol"
    transmission: str = "Automatic"
    condition: str = "Registered (Used)"
    location: str = "Unknown"


class PredictResponse(BaseModel):
    predicted_price: float
    range_low: float
    range_high: float
    currency: str
    inputs_used: dict


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalize_condition(raw: str) -> str:
    s = raw.lower()
    if "recondition" in s:
        return "Reconditioned"
    if "unregister" in s:
        return "Unregistered"
    if "brand new" in s or "brandnew" in s:
        return "Brand New"
    if "used" in s:
        return "Used"
    return "Other"


def build_input(req: PredictRequest) -> pd.DataFrame:
    row: dict = {
        "year":        req.year,
        "age":         CURRENT_YEAR - req.year,
        "log_mileage": np.log1p(req.mileage),
    }

    for col, val in [("make", req.make), ("location", req.location)]:
        if col in bundle["label_encoders"]:
            le = bundle["label_encoders"][col]
            try:
                row[col] = int(le.transform([val])[0])
            except ValueError:
                log.warning(f"Unseen {col} value '{val}' — encoding as -1")
                row[col] = -1

    for feat in bundle["feature_cols"]:
        if feat not in row:
            row[feat] = 0

    for prefix, value in [("fuel_type", req.fuel_type), ("transmission", req.transmission), ("condition", _normalize_condition(req.condition))]:
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1

    return pd.DataFrame([row])[bundle["feature_cols"]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def health():
    return {"status": "ok", "model_loaded": bundle is not None}


@app.get("/makes")
async def makes():
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    le = bundle["label_encoders"].get("make")
    return {"makes": le.classes_.tolist() if le is not None else []}


@app.get("/market-stats")
async def market_stats():
    path = "data/processed.csv"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="processed.csv not found — run preprocess.py first")

    df = pd.read_csv(path)
    price = df["price"]

    top_makes = []
    if bundle is not None and "make" in df.columns and "make" in bundle["label_encoders"]:
        try:
            le = bundle["label_encoders"]["make"]
            make_names = le.inverse_transform(df["make"].astype(int))
            counts = pd.Series(make_names).value_counts().head(5)
            top_makes = [{"make": str(m), "count": int(c)} for m, c in counts.items()]
        except Exception as e:
            log.warning(f"Could not decode makes: {e}")

    fuel_cols = [c for c in df.columns if c.startswith("fuel_type_")]
    fuel_dist = {
        col.replace("fuel_type_", ""): int(df[col].astype(bool).sum())
        for col in fuel_cols
    }

    return {
        "total_listings":  int(len(df)),
        "avg_price":       float(price.mean()),
        "median_price":    float(price.median()),
        "min_price":       float(price.min()),
        "max_price":       float(price.max()),
        "top_makes":       top_makes,
        "fuel_distribution": fuel_dist,
        "currency":        "LKR",
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run train.py first")

    X = build_input(req)
    price = float(np.exp(bundle["model"].predict(X)[0]))

    return PredictResponse(
        predicted_price=price,
        range_low=price * 0.88,
        range_high=price * 1.12,
        currency="LKR",
        inputs_used=req.model_dump(),
    )
