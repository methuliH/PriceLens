import os
import logging
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from db.mongo import listings as listings_col

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_PATH = "models/price_model.joblib"
CURRENT_YEAR = 2026

bundle: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    load_dotenv()
    if not os.path.exists(MODEL_PATH):
        log.warning(f"Model not found at {MODEL_PATH} — run ml/train.py first. API will start without it.")
    else:
        try:
            bundle = joblib.load(MODEL_PATH)
            log.info(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            log.warning(f"Failed to load model: {e}")
    yield


app = FastAPI(title="PriceLens", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    make: str
    model: str | None = None
    year: int = Field(..., ge=1980, le=2026)
    mileage: float = Field(..., ge=0)
    fuel_type: str = "Petrol"
    transmission: str = "Automatic"
    condition: str = "Registered (Used)"
    location: str = "Unknown"


class ComparableListing(BaseModel):
    title: str | None
    make: str | None
    year: int | None
    mileage: float | None
    price: float
    fuel_type: str | None
    transmission: str | None
    condition: str | None
    url: str | None


class PredictResponse(BaseModel):
    predicted_price: float
    range_low: float
    range_high: float
    currency: str
    inputs_used: dict
    comparables: list[ComparableListing]


class ListingOut(BaseModel):
    title: str | None
    make: str | None
    model: str | None
    year: int | None
    mileage: float | None
    engine_cc: int | None
    price: float | None
    fuel_type: str | None
    transmission: str | None
    condition: str | None
    location: str | None
    url: str | None


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
    model_str = (req.model or "Unknown").strip() or "Unknown"

    row: dict = {
        "year":        req.year,
        "age":         CURRENT_YEAR - req.year,
        "log_mileage": np.log1p(req.mileage),
    }

    # make_model_mean_price from saved lookup; fall back to global mean if unseen
    if "make_model_means" in bundle:
        key = (req.make, model_str)
        row["make_model_mean_price"] = bundle["make_model_means"].get(
            key, bundle.get("global_mean_price", 0.0)
        )

    # Label-encode make, model, location
    for col, val in [("make", req.make), ("model", model_str), ("location", req.location)]:
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

    for prefix, value in [
        ("fuel_type", req.fuel_type),
        ("transmission", req.transmission),
        ("condition", _normalize_condition(req.condition)),
    ]:
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
    col = listings_col()
    db_makes = await col.distinct("make", {"is_active": True, "make": {"$ne": None}})
    return {"makes": sorted(m for m in db_makes if m)}


@app.get("/models/{make}")
async def models(make: str):
    col = listings_col()
    db_models = await col.distinct("model", {"is_active": True, "make": make, "model": {"$ne": None}})
    return {"models": sorted(m for m in db_models if m)}


@app.get("/market-stats")
async def market_stats():
    col = listings_col()
    match = {"is_active": True, "price": {"$ne": None}}

    pipeline = [
        {"$match": match},
        {"$facet": {
            "stats": [{"$group": {
                "_id":       None,
                "count":     {"$sum": 1},
                "avg_price": {"$avg": "$price"},
                "min_price": {"$min": "$price"},
                "max_price": {"$max": "$price"},
            }}],
            "top_makes": [
                {"$match": {"make": {"$ne": None}}},
                {"$group": {"_id": "$make", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 5},
                {"$project": {"_id": 0, "make": "$_id", "count": 1}},
            ],
            "fuel_dist": [
                {"$match": {"fuel_type": {"$ne": None}}},
                {"$group": {"_id": "$fuel_type", "count": {"$sum": 1}}},
                {"$project": {"_id": 0, "fuel_type": "$_id", "count": 1}},
            ],
        }},
    ]
    agg_result = await col.aggregate(pipeline).to_list(length=1)

    if not agg_result or not agg_result[0]["stats"]:
        raise HTTPException(status_code=404, detail="No active listings in database")

    facet = agg_result[0]
    stats = facet["stats"][0]

    price_docs = await col.find(match, {"price": 1, "_id": 0}).to_list(length=10_000)
    prices = np.array([d["price"] for d in price_docs])

    fuel_distribution = {entry["fuel_type"]: entry["count"] for entry in facet["fuel_dist"]}

    return {
        "total_listings":    int(stats["count"]),
        "avg_price":         float(stats["avg_price"]),
        "median_price":      float(np.median(prices)),
        "min_price":         float(stats["min_price"]),
        "max_price":         float(stats["max_price"]),
        "top_makes":         facet["top_makes"],
        "fuel_distribution": fuel_distribution,
        "currency":          "LKR",
    }


@app.get("/listings", response_model=list[ListingOut])
async def listings(
    make: str | None = Query(None),
    model: str | None = Query(None),
    fuel_type: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    sort: Literal["price_asc", "price_desc", "newest"] = Query("newest"),
):
    col = listings_col()
    query: dict = {"is_active": True}
    if make:
        query["make"] = make
    if model:
        query["model"] = model
    if fuel_type:
        query["fuel_type"] = fuel_type
    if sort in ("price_asc", "price_desc"):
        query["price"] = {"$ne": None}

    sort_field, sort_dir = {
        "price_asc":  ("price", 1),
        "price_desc": ("price", -1),
        "newest":     ("scraped_at", -1),
    }[sort]

    docs = (
        await col.find(query, {"_id": 0})
        .sort(sort_field, sort_dir)
        .limit(limit)
        .to_list(length=limit)
    )
    return [
        ListingOut(
            title=d.get("title"),
            make=d.get("make"),
            model=d.get("model"),
            year=d.get("year"),
            mileage=d.get("mileage"),
            engine_cc=d.get("engine_cc"),
            price=d.get("price"),
            fuel_type=d.get("fuel_type"),
            transmission=d.get("transmission"),
            condition=d.get("condition"),
            location=d.get("location"),
            url=d.get("url"),
        )
        for d in docs
    ]


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded — run ml/train.py first")

    X = build_input(req)
    raw_pred = float(bundle["model"].predict(X)[0])
    price = float(np.expm1(raw_pred) if bundle.get("log_transform") else raw_pred)

    col = listings_col()
    raw_comps = await (
        col.find(
            {"is_active": True, "make": req.make, "price": {"$ne": None}},
            {"title": 1, "make": 1, "year": 1, "mileage": 1, "price": 1,
             "fuel_type": 1, "transmission": 1, "condition": 1, "url": 1, "_id": 0},
        )
        .sort("scraped_at", -1)
        .limit(5)
        .to_list(length=5)
    )

    comparables = [
        ComparableListing(
            title=d.get("title"),
            make=d.get("make"),
            year=d.get("year"),
            mileage=d.get("mileage"),
            price=d["price"],
            fuel_type=d.get("fuel_type"),
            transmission=d.get("transmission"),
            condition=d.get("condition"),
            url=d.get("url"),
        )
        for d in raw_comps
    ]

    return PredictResponse(
        predicted_price=price,
        range_low=price * 0.88,
        range_high=price * 1.12,
        currency="LKR",
        inputs_used={**req.model_dump(), "make_model_mean_price": float(X["make_model_mean_price"].iloc[0]) if "make_model_mean_price" in X.columns else None},
        comparables=comparables,
    )
