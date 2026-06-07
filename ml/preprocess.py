"""
preprocess.py
-------------
Cleans raw scraped CSV and engineers features for model training.

Usage:
    python preprocess.py --input data/raw_listings.csv --output data/processed.csv
"""

import os
import pandas as pd
import numpy as np
import re
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CURRENT_YEAR = 2026

# Common Sri Lankan vehicle makes — used to extract make from title
KNOWN_MAKES = [
    "Toyota", "Honda", "Suzuki", "Nissan", "Mitsubishi", "Mazda", "Hyundai",
    "Kia", "BMW", "Mercedes", "Audi", "Volkswagen", "Perodua", "Maruti",
    "Ford", "Subaru", "Isuzu", "Tata", "Land Rover", "Jeep", "Lexus",
    "Peugeot", "Renault", "Volvo", "Skoda", "Daihatsu", "Bajaj"
]


# ── Parsing helpers ──────────────────────────────────────────────────────────

def parse_price(raw: str) -> float | None:
    """Convert price string like 'Rs. 4,500,000' → 4500000.0"""
    if not isinstance(raw, str):
        return None
    # Remove everything except digits (Sri Lankan prices have no decimal component)
    cleaned = re.sub(r"[^\d]", "", raw)
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_mileage(raw: str) -> float | None:
    """Convert mileage string like '85,000 km' → 85000.0"""
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[^\d]", "", raw)
    try:
        val = float(cleaned)
        return val if val < 2_000_000 else None  # sanity cap
    except ValueError:
        return None


def extract_year(text: str) -> int | None:
    """Find a 4-digit year (1980–2026) in a string."""
    if not isinstance(text, str):
        return None
    matches = re.findall(r"\b(19[89]\d|20[012]\d)\b", text)
    return int(matches[0]) if matches else None


def extract_make(title: str) -> str | None:
    """Try to match a known make from the listing title."""
    if not isinstance(title, str):
        return None
    title_lower = title.lower()
    for make in KNOWN_MAKES:
        if make.lower() in title_lower:
            return make
    return None


def fuel_from_title(title: str) -> str:
    """Guess fuel type from vehicle model name."""
    if not isinstance(title, str):
        return "Petrol"
    t = title.lower()
    electric_models = ["leaf", "tesla", "bolt", "ioniq electric", "zoe", "e-golf", "bz4x"]
    hybrid_models   = ["prius", "aqua", "insight", "grace", "freed hybrid", "fit hybrid",
                       "axio hybrid", "allion hybrid", "vezel hybrid", "nbox", "spade",
                       "shuttle", "jade", "stepwgn spada", "odyssey hybrid", "cr-z",
                       "camry hybrid", "harrier hybrid", "alphard hybrid", "vellfire hybrid"]
    diesel_models   = ["hilux", "land cruiser", "kdh", "defender", "pajero", "montero",
                       "d-max", "ranger", "triton", "l200", "bt-50", "pick up", "pickup",
                       "4runner", "fortuner", "prado", "surf"]
    if any(m in t for m in electric_models) or "electric" in t:
        return "Electric"
    if any(m in t for m in hybrid_models) or "hybrid" in t:
        return "Hybrid"
    if any(m in t for m in diesel_models) or "diesel" in t:
        return "Diesel"
    # Default: petrol — the vast majority of Sri Lankan used cars
    return "Petrol"


def transmission_from_title(title: str) -> str:
    """Guess transmission from title keywords."""
    if not isinstance(title, str):
        return "Automatic"
    t = title.lower()
    manual_keywords = ["manual", " mt ", "5speed", "6speed", "5-speed", "6-speed", "gear"]
    auto_models     = ["prius", "aqua", "leaf", "axio", "allion", "premio", "vezel",
                       "fit", "grace", "freed", "spade", "wish", "voxy", "noah",
                       "alphard", "vellfire", "harrier", "rav4", "crv", "hrv",
                       "civic", "accord", "camry", "corolla", "belta", "yaris",
                       "wagon r", "alto", "swift", "hustler", "every", "tanto",
                       "march", "note", "serena", "elgrand", "x-trail", "murano",
                       "outlander", "lancer", "evo", "asx"]
    if any(k in t for k in manual_keywords):
        return "Manual"
    if any(m in t for m in auto_models) or "automatic" in t or "auto" in t:
        return "Automatic"
    # Default: automatic — majority of Japanese imports in Sri Lanka
    return "Automatic"


def normalize_fuel(raw: str) -> str:
    """Normalize fuel type strings to clean categories."""
    if not isinstance(raw, str):
        return "Unknown"
    raw = raw.lower()
    if "petrol" in raw:
        return "Petrol"
    if "diesel" in raw:
        return "Diesel"
    if "electric" in raw:
        return "Electric"
    if "hybrid" in raw:
        return "Hybrid"
    if "gas" in raw or "lpg" in raw or "cng" in raw:
        return "Gas"
    return "Other"


def normalize_transmission(raw: str) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    raw = raw.lower()
    if "auto" in raw or "tiptronic" in raw or "cvt" in raw:
        return "Automatic"
    if "manual" in raw:
        return "Manual"
    return "Other"


def normalize_condition(raw: str) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    raw = raw.lower()
    if "recondition" in raw:
        return "Reconditioned"
    if "unregister" in raw:
        return "Unregistered"
    if "brand new" in raw or "brandnew" in raw:
        return "Brand New"
    if "used" in raw:
        return "Used"
    return "Other"


# ── Main pipeline ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Input shape: {df.shape}")

    # ── Price ──
    df["price"] = df["price_raw"].apply(parse_price)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 100_000]         # remove obviously wrong prices
    df = df[df["price"] < 100_000_000]     # remove outliers

    # ── Mileage ──
    df["mileage"] = df["mileage_raw"].apply(parse_mileage)

    # ── Year ── (try dedicated column first, fallback to title)
    if "year" in df.columns:
        df["year"] = df["year"].apply(lambda x: extract_year(str(x)))
    else:
        df["year"] = df["title"].apply(extract_year)

    df = df[df["year"].between(1980, CURRENT_YEAR, inclusive="both")]

    # ── Derived: vehicle age ──
    df["age"] = CURRENT_YEAR - df["year"]

    # ── Make ── (try dedicated column first, fallback to title)
    if "make" not in df.columns:
        df["make"] = df["title"].apply(extract_make)

    # ── Normalize categoricals (may be missing if --no-details was used) ──
    if "fuel_type" not in df.columns:
        df["fuel_type"] = "Unknown"
    if "transmission" not in df.columns:
        df["transmission"] = "Unknown"
    if "condition" not in df.columns:
        df["condition"] = "Unknown"

    df["fuel_type"]    = df["fuel_type"].apply(normalize_fuel)
    df["transmission"] = df["transmission"].apply(normalize_transmission)
    df["condition"]    = df["condition"].apply(normalize_condition)

    # Fill unknowns using title inference
    mask_fuel = df["fuel_type"].isin(["Unknown", "Other"])
    df.loc[mask_fuel, "fuel_type"] = df.loc[mask_fuel, "title"].apply(fuel_from_title)

    mask_trans = df["transmission"].isin(["Unknown", "Other"])
    df.loc[mask_trans, "transmission"] = df.loc[mask_trans, "title"].apply(transmission_from_title)

    # ── Drop rows with too many missing features ──
    core_features = ["price", "year", "mileage", "fuel_type", "transmission"]
    df = df.dropna(subset=["mileage"])  # mileage is critical

    # ── Log-transform mileage (reduces skew) ──
    df["log_mileage"] = np.log1p(df["mileage"])

    # ── Drop duplicates ──
    df = df.drop_duplicates(subset=["title", "price", "mileage"], keep="first")

    # ── Keep only useful columns ──
    keep_cols = [
        "title", "make", "year", "age", "mileage", "log_mileage",
        "fuel_type", "transmission", "condition", "location", "price", "url"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].reset_index(drop=True)

    log.info(f"After cleaning: {df.shape}")
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns for model training."""
    cat_cols = ["fuel_type", "transmission", "condition"]
    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns], drop_first=False)
    return df


# ── MongoDB / CSV loaders ─────────────────────────────────────────────────────

def load_from_mongo() -> pd.DataFrame:
    """Read active listings from MongoDB, clean and OHE-encode them.

    Returns a DataFrame in the same format as processed.csv, with an extra
    `model` string column ready for label encoding in train.py.
    Raises RuntimeError if MONGO_URI is not set.
    """
    from pymongo import MongoClient
    from dotenv import load_dotenv
    load_dotenv()

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set — add it to .env or environment")
    db_name = os.getenv("DB_NAME", "pricelens")

    log.info("Loading data from MongoDB...")
    client = MongoClient(uri)
    col    = client[db_name]["listings"]
    docs   = list(col.find(
        {"is_active": True, "price": {"$ne": None}},
        {"_id": 0, "title": 1, "make": 1, "model": 1, "year": 1, "mileage": 1,
         "price": 1, "fuel_type": 1, "transmission": 1, "condition": 1,
         "location": 1, "url": 1},
    ))
    client.close()
    log.info(f"Fetched {len(docs)} documents from MongoDB")

    if not docs:
        raise RuntimeError("No active listings with price found in MongoDB")

    df = pd.DataFrame(docs)

    # Numeric — already parsed in the DB; coerce and drop bad rows
    df["price"]   = pd.to_numeric(df["price"],   errors="coerce")
    df["mileage"] = pd.to_numeric(df["mileage"] if "mileage" in df.columns else np.nan, errors="coerce")
    df["year"]    = pd.to_numeric(df["year"]    if "year"    in df.columns else np.nan, errors="coerce")

    df = df.dropna(subset=["price", "mileage", "year"])
    df = df[(df["price"] > 100_000) & (df["price"] < 100_000_000)]
    df = df[df["mileage"] < 2_000_000]
    df["year"] = df["year"].astype(int)
    df = df[df["year"].between(1980, CURRENT_YEAR)]

    # Derived features
    df["age"]         = CURRENT_YEAR - df["year"]
    df["log_mileage"] = np.log1p(df["mileage"])

    # Normalize categoricals
    for col_name in ["fuel_type", "transmission", "condition"]:
        if col_name not in df.columns:
            df[col_name] = "Unknown"
    df["fuel_type"]    = df["fuel_type"].apply(normalize_fuel)
    df["transmission"] = df["transmission"].apply(normalize_transmission)
    df["condition"]    = df["condition"].apply(normalize_condition)

    # Infer still-unknown fuel/transmission from title
    if "title" in df.columns:
        mask_fuel  = df["fuel_type"].isin(["Unknown", "Other"])
        mask_trans = df["transmission"].isin(["Unknown", "Other"])
        df.loc[mask_fuel,  "fuel_type"]    = df.loc[mask_fuel,  "title"].apply(fuel_from_title)
        df.loc[mask_trans, "transmission"] = df.loc[mask_trans, "title"].apply(transmission_from_title)

    # model — keep as string; label encoding happens in train.py
    if "model" not in df.columns:
        df["model"] = "Unknown"
    df["model"] = df["model"].fillna("Unknown").astype(str)

    # Remaining string fields
    df["make"]     = df["make"].fillna("Unknown").astype(str)     if "make"     in df.columns else "Unknown"
    df["location"] = df["location"].fillna("Unknown").astype(str) if "location" in df.columns else "Unknown"

    # Drop duplicates
    df = df.drop_duplicates(subset=["title", "price", "mileage"], keep="first")

    keep = ["title", "make", "model", "year", "age", "mileage", "log_mileage",
            "fuel_type", "transmission", "condition", "location", "price", "url"]
    df = df[[c for c in keep if c in df.columns]].reset_index(drop=True)
    log.info(f"After cleaning: {df.shape}")

    return encode(df)


def load_from_csv(path: str = "data/processed.csv") -> pd.DataFrame:
    """Load an already-processed CSV (output of the preprocess pipeline)."""
    df = pd.read_csv(path)
    log.info(f"Loaded from CSV: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw_listings.csv")
    parser.add_argument("--output", default="data/processed.csv")
    args = parser.parse_args()

    df_raw = pd.read_csv(args.input)
    log.info(f"Loaded raw data: {df_raw.shape}")

    df_clean = clean(df_raw)
    df_encoded = encode(df_clean)

    df_encoded.to_csv(args.output, index=False)
    log.info(f"Saved processed data to {args.output}")

    # Quick summary
    print("\n── Price distribution (LKR) ──")
    print(df_clean["price"].describe().apply(lambda x: f"{x:,.0f}"))
    print("\n── Fuel types ──")
    print(df_clean["fuel_type"].value_counts())
    print("\n── Transmission ──")
    print(df_clean["transmission"].value_counts())


if __name__ == "__main__":
    main()