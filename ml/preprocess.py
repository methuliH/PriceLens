"""
preprocess.py
-------------
Cleans raw scraped CSV and engineers features for model training.

Usage:
    python preprocess.py --input data/raw_listings.csv --output data/processed.csv
"""

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CURRENT_YEAR = 2026

KNOWN_MAKES = [
    "Toyota", "Honda", "Suzuki", "Nissan", "Mitsubishi", "Mazda", "Hyundai",
    "Kia", "BMW", "Mercedes", "Audi", "Volkswagen", "Perodua", "Maruti",
    "Ford", "Subaru", "Isuzu", "Tata", "Land Rover", "Jeep", "Lexus",
    "Peugeot", "Renault", "Volvo", "Skoda", "Daihatsu", "Bajaj"
]


# ── Location normalisation ────────────────────────────────────────────────────

LOCATION_MAP: dict[str, str] = {
    "Ambalangoda":   "Galle",
    "Ampara":        "Ampara",
    "Anuradapura":   "Anuradhapura",
    "Anuradhapura":  "Anuradhapura",
    "Avissawella":   "Colombo",
    "Badulla":       "Badulla",
    "Balangoda":     "Ratnapura",
    "Bandaragama":   "Kalutara",
    "Bandarawela":   "Badulla",
    "Battaramulla":  "Colombo",
    "Batticaloa":    "Batticaloa",
    "Bentota":       "Galle",
    "Beruwala":      "Kalutara",
    "Boralesgamuwa": "Colombo",
    "Chilaw":        "Puttalam",
    "Colombo":       "Colombo",
    **{f"Colombo {i}": "Colombo" for i in range(1, 16)},
    "Col":           "Colombo",
    "Dambulla":      "Matale",
    "Dankotuwa":     "Puttalam",
    "Dehiwala":      "Colombo",
    "Dehiwala-Mt":   "Colombo",
    "Mount Lavinia": "Colombo",
    "Divulapitiya":  "Gampaha",
    "Dompe":         "Gampaha",
    "Eheliyagoda":   "Ratnapura",
    "Elpitiya":      "Galle",
    "Embilipitiya":  "Ratnapura",
    "Eravur":        "Batticaloa",
    "Galle":         "Galle",
    "Gampaha":       "Gampaha",
    "Gampola":       "Kandy",
    "Ganemulla":     "Gampaha",
    "Hambantota":    "Hambantota",
    "Hanwella":      "Colombo",
    "Harispattuwa":  "Kandy",
    "Hatton":        "Nuwara Eliya",
    "Hendala":       "Gampaha",
    "Homagama":      "Colombo",
    "Horana":        "Kalutara",
    "Ja-Ela":        "Gampaha",
    "Jaffna":        "Jaffna",
    "Kadawatha":     "Gampaha",
    "Kadugannawa":   "Kandy",
    "Kaduwela":      "Colombo",
    "Kalmunai":      "Ampara",
    "Kalutara":      "Kalutara",
    "Kandana":       "Gampaha",
    "Kandy":         "Kandy",
    "Katunayake":    "Gampaha",
    "Kegalle":       "Kegalle",
    "Kelaniya":      "Gampaha",
    "Kesbewa":       "Colombo",
    "Kilinochchi":   "Kilinochchi",
    "Kiribathgoda":  "Gampaha",
    "Koggala":       "Galle",
    "Kolonnawa":     "Colombo",
    "Kotikawatta":   "Colombo",
    "Kottawa":       "Colombo",
    "Kotte":         "Colombo",
    "Kuliyapitiya":  "Kurunegala",
    "Kurunegala":    "Kurunegala",
    "Maharagama":    "Colombo",
    "Mahiyanganaya": "Badulla",
    "Malabe":        "Colombo",
    "Marawila":      "Puttalam",
    "Matale":        "Matale",
    "Matara":        "Matara",
    "Matugama":      "Kalutara",
    "Mawanella":     "Kegalle",
    "Middeniya":     "Hambantota",
    "Minuwangoda":   "Gampaha",
    "Mirigama":      "Gampaha",
    "Moneragala":    "Moneragala",
    "Moratuwa":      "Colombo",
    "Mulleriyawa":   "Colombo",
    "Nattandiya":    "Puttalam",
    "Nawala":        "Colombo",
    "Nawalapitiya":  "Kandy",
    "Negombo":       "Gampaha",
    "Nittambuwa":    "Gampaha",
    "Nugegoda":      "Colombo",
    "Nuwara-Eliya":  "Nuwara Eliya",
    "Nuwara Eliya":  "Nuwara Eliya",
    "Padukka":       "Colombo",
    "Panadura":      "Kalutara",
    "Pannala":       "Kurunegala",
    "Pannipitiya":   "Colombo",
    "Piliyandala":   "Colombo",
    "Polgahawela":   "Kurunegala",
    "Polonnaruwa":   "Polonnaruwa",
    "Puttalam":      "Puttalam",
    "Ragama":        "Gampaha",
    "Rajagiriya":    "Colombo",
    "Ratnapura":     "Ratnapura",
    "Tangalle":      "Hambantota",
    "Trincomalee":   "Trincomalee",
    "Vavuniya":      "Vavuniya",
    "Veyangoda":     "Gampaha",
    "Warakapola":    "Kegalle",
    "Wattala":       "Gampaha",
    "Wattegama":     "Kandy",
    "Weligama":      "Matara",
    "Welimada":      "Badulla",
    "Welisara":      "Gampaha",
    "Wennappuwa":    "Puttalam",
}


# ── Parsing helpers ──────────────────────────────────────────────────────────

def parse_price(raw: str) -> float | None:
    """Convert price string like 'Rs. 4,500,000' → 4500000.0"""
    if not isinstance(raw, str):
        return None
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
        return val if val < 2_000_000 else None
    except ValueError:
        return None


def extract_year(text: str) -> int | None:
    """Find a 4-digit year (1970–2026) in a string."""
    if not isinstance(text, str):
        return None
    matches = re.findall(r"\b(19[789]\d|20[012]\d)\b", text)
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


def normalize_location(raw: str) -> str:
    """Map a city/suburb name to its canonical Sri Lanka district name."""
    if not isinstance(raw, str) or not raw.strip():
        return "Unknown"
    val = raw.strip()
    if val in LOCATION_MAP:
        return LOCATION_MAP[val]
    if val.lower().startswith("col"):
        return "Colombo"
    return val


# ── Data quality helpers ──────────────────────────────────────────────────────

def _filter_price_iqr(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop price outliers per make using 1.5×IQR. Groups with < 4 rows are kept as-is."""
    before = len(df)
    parts = []
    for _make, grp in df.groupby("make", dropna=False):
        if len(grp) < 4:
            parts.append(grp)
            continue
        q1 = grp["price"].quantile(0.25)
        q3 = grp["price"].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        parts.append(grp[grp["price"].between(lo, hi)])
    result = pd.concat(parts).reset_index(drop=True) if parts else df.iloc[0:0].copy()
    return result, before - len(result)


def _impute_mileage(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Impute null mileage with the median for (make, year). Falls back to global median."""
    null_mask = df["mileage"].isna()
    n_null = int(null_mask.sum())
    if n_null == 0:
        return df, 0

    medians: dict = (
        df[~null_mask]
        .groupby(["make", "year"], dropna=False)["mileage"]
        .median()
        .to_dict()
    )
    global_median = float(df[~null_mask]["mileage"].median())

    def _fill(row):
        if pd.isna(row["mileage"]):
            return medians.get((row["make"], row["year"]), global_median)
        return row["mileage"]

    df = df.copy()
    df["mileage"] = df.apply(_fill, axis=1)
    return df, n_null


def _dedup_temporal(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop duplicates where make/model/year/price/location all match and scraped_at
    falls in the same 30-day window. Keeps the most recently scraped copy.
    Falls back to title/price/mileage dedup when scraped_at is absent.
    """
    before = len(df)
    KEY = ["make", "model", "year", "price", "location"]
    key_cols = [c for c in KEY if c in df.columns]

    if "scraped_at" in df.columns:
        df = df.sort_values("scraped_at", ascending=False, na_position="last").copy()
        df["_ts"] = pd.to_datetime(df["scraped_at"], utc=True, errors="coerce")
        df["_bucket"] = (df["_ts"].astype("int64") // (30 * 24 * 3600 * 10**9))
        df = df.drop_duplicates(subset=key_cols + ["_bucket"], keep="first")
        df = df.drop(columns=["_ts", "_bucket"])
    else:
        df = df.drop_duplicates(subset=key_cols, keep="first")

    return df.reset_index(drop=True), before - len(df)


def _clean_mongo_df(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply all data quality steps and return (cleaned_df, report)."""
    report: dict[str, int] = {}
    start = len(df)

    # 1 — Price hard bounds
    before = len(df)
    df = df[(df["price"] > 100_000) & (df["price"] < 100_000_000)]
    report["price_hard_bounds_dropped"] = before - len(df)

    # 2 — Price IQR per make
    df, n = _filter_price_iqr(df)
    report["price_iqr_dropped"] = n

    # 3 — Mileage: zero → null, > 1 000 000 → drop
    df = df.copy()
    zero_mask = df["mileage"] == 0
    report["mileage_zero_nulled"] = int(zero_mask.sum())
    df.loc[zero_mask, "mileage"] = np.nan
    before = len(df)
    df = df[df["mileage"].isna() | (df["mileage"] <= 1_000_000)]
    report["mileage_over_1m_dropped"] = before - len(df)

    # 4 — Mileage imputation
    df, n = _impute_mileage(df)
    report["mileage_imputed"] = n

    # 5 — Year range (1970 – CURRENT_YEAR)
    before = len(df)
    df = df[df["year"].between(1970, CURRENT_YEAR)]
    report["year_out_of_range_dropped"] = before - len(df)

    # 6 — Temporal duplicate removal
    df, n = _dedup_temporal(df)
    report["duplicates_dropped"] = n

    # 7 — Location normalisation
    if "location" in df.columns:
        before_locs = df["location"].copy()
        df["location"] = df["location"].apply(normalize_location)
        report["locations_normalised"] = int((df["location"] != before_locs.fillna("")).sum())

    report["total_dropped"] = start - len(df)
    report["rows_after"] = len(df)
    return df.reset_index(drop=True), report


def _print_report(start: int, report: dict) -> None:
    print("\n" + "=" * 52)
    print("  DATA QUALITY REPORT")
    print("=" * 52)
    print(f"  Input rows                    : {start:>6}")
    print(f"  Price hard-bounds dropped     : {report.get('price_hard_bounds_dropped', 0):>6}")
    print(f"  Price IQR outliers dropped    : {report.get('price_iqr_dropped', 0):>6}")
    print(f"  Mileage == 0 -> null          : {report.get('mileage_zero_nulled', 0):>6}")
    print(f"  Mileage > 1 000 000 dropped   : {report.get('mileage_over_1m_dropped', 0):>6}")
    print(f"  Mileage nulls imputed         : {report.get('mileage_imputed', 0):>6}")
    print(f"  Year out-of-range dropped     : {report.get('year_out_of_range_dropped', 0):>6}")
    print(f"  Duplicates dropped            : {report.get('duplicates_dropped', 0):>6}")
    print(f"  Locations normalised          : {report.get('locations_normalised', 0):>6}")
    print(f"  ------------------------------------")
    print(f"  Total rows dropped            : {report.get('total_dropped', 0):>6}")
    print(f"  Rows remaining                : {report.get('rows_after', 0):>6}")
    print("=" * 52 + "\n")


# ── Main pipeline ────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Input shape: {df.shape}")

    df["price"] = df["price_raw"].apply(parse_price)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 100_000]
    df = df[df["price"] < 100_000_000]

    df["mileage"] = df["mileage_raw"].apply(parse_mileage)

    if "year" in df.columns:
        df["year"] = df["year"].apply(lambda x: extract_year(str(x)))
    else:
        df["year"] = df["title"].apply(extract_year)

    df = df[df["year"].between(1970, CURRENT_YEAR, inclusive="both")]
    df["age"] = CURRENT_YEAR - df["year"]

    if "make" not in df.columns:
        df["make"] = df["title"].apply(extract_make)

    for col_name in ["fuel_type", "transmission", "condition"]:
        if col_name not in df.columns:
            df[col_name] = "Unknown"

    df["fuel_type"]    = df["fuel_type"].apply(normalize_fuel)
    df["transmission"] = df["transmission"].apply(normalize_transmission)
    df["condition"]    = df["condition"].apply(normalize_condition)

    mask_fuel = df["fuel_type"].isin(["Unknown", "Other"])
    df.loc[mask_fuel, "fuel_type"] = df.loc[mask_fuel, "title"].apply(fuel_from_title)

    mask_trans = df["transmission"].isin(["Unknown", "Other"])
    df.loc[mask_trans, "transmission"] = df.loc[mask_trans, "title"].apply(transmission_from_title)

    if "location" in df.columns:
        df["location"] = df["location"].apply(normalize_location)

    df = df.dropna(subset=["mileage"])
    df["log_mileage"] = np.log1p(df["mileage"])
    df = df.drop_duplicates(subset=["title", "price", "mileage"], keep="first")

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
    """Read active listings from MongoDB, apply data quality pipeline, OHE-encode.

    Returns a DataFrame in the same format as processed.csv with an extra
    `model` string column ready for label encoding in train.py.
    Raises RuntimeError if MONGO_URI is not set.
    """
    from dotenv import load_dotenv
    from pymongo import MongoClient

    load_dotenv()

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set — add it to .env or environment")
    db_name = os.getenv("MONGO_DB", "price_prediction")

    log.info("Loading data from MongoDB...")
    client = MongoClient(uri)
    col    = client[db_name]["listings"]
    docs   = list(col.find(
        {"is_active": True, "price": {"$ne": None}},
        {"_id": 0, "title": 1, "make": 1, "model": 1, "year": 1, "mileage": 1,
         "price": 1, "fuel_type": 1, "transmission": 1, "condition": 1,
         "location": 1, "url": 1, "scraped_at": 1},
    ))
    client.close()
    log.info(f"Fetched {len(docs)} documents from MongoDB")

    if not docs:
        raise RuntimeError("No active listings with price found in MongoDB")

    df = pd.DataFrame(docs)

    df["price"]   = pd.to_numeric(df["price"],   errors="coerce")
    df["mileage"] = pd.to_numeric(df.get("mileage"), errors="coerce")
    df["year"]    = pd.to_numeric(df.get("year"),    errors="coerce")
    df = df.dropna(subset=["price", "year"])
    df["year"] = df["year"].astype(int)
    df["age"]  = CURRENT_YEAR - df["year"]

    for col_name in ["fuel_type", "transmission", "condition"]:
        if col_name not in df.columns:
            df[col_name] = "Unknown"
    df["fuel_type"]    = df["fuel_type"].apply(normalize_fuel)
    df["transmission"] = df["transmission"].apply(normalize_transmission)
    df["condition"]    = df["condition"].apply(normalize_condition)

    if "title" in df.columns:
        mask_fuel  = df["fuel_type"].isin(["Unknown", "Other"])
        mask_trans = df["transmission"].isin(["Unknown", "Other"])
        df.loc[mask_fuel,  "fuel_type"]    = df.loc[mask_fuel,  "title"].apply(fuel_from_title)
        df.loc[mask_trans, "transmission"] = df.loc[mask_trans, "title"].apply(transmission_from_title)

    if "model" not in df.columns:
        df["model"] = "Unknown"
    df["model"]    = df["model"].fillna("Unknown").astype(str)
    df["make"]     = df["make"].fillna("Unknown").astype(str)     if "make"     in df.columns else "Unknown"
    df["location"] = df["location"].fillna("Unknown").astype(str) if "location" in df.columns else "Unknown"

    # ── Data quality pipeline ─────────────────────────────────────────────────
    start = len(df)
    df, report = _clean_mongo_df(df)
    _print_report(start, report)

    df["log_mileage"] = np.log1p(df["mileage"])

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

    print("\n── Price distribution (LKR) ──")
    print(df_clean["price"].describe().apply(lambda x: f"{x:,.0f}"))
    print("\n── Fuel types ──")
    print(df_clean["fuel_type"].value_counts())
    print("\n── Transmission ──")
    print(df_clean["transmission"].value_counts())


if __name__ == "__main__":
    main()
