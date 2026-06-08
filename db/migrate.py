"""
migrate.py — load raw_listings.csv into MongoDB (pricelens DB)
Run once: python db/migrate.py
"""

import os
import re
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

MONGO_URI    = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME      = os.getenv("DB_NAME", "pricelens")
CURRENT_YEAR = 2026

# ── helpers ────────────────────────────────────────────────────────────────────

def parse_price(raw) -> float | None:
    if pd.isna(raw):
        return None
    # Match the first run of digits+commas (e.g. "Rs. 6,000,000" → "6,000,000")
    m = re.search(r"[\d,]+(?:\.\d+)?", str(raw))
    if not m:
        return None
    return float(m.group().replace(",", "")) or None


def parse_year(raw) -> int | None:
    if pd.isna(raw):
        return None
    m = re.search(r"\d{4}", str(raw))
    return int(m.group()) if m else None


def parse_mileage(raw) -> float | None:
    if pd.isna(raw):
        return None
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    return float(cleaned) if cleaned else None


def parse_engine_cc(raw) -> int | None:
    if pd.isna(raw):
        return None
    cleaned = re.sub(r"[^\d]", "", str(raw))
    return int(cleaned) if cleaned else None


def extract_make(title: str, known_makes: list[str]) -> str | None:
    if pd.isna(title):
        return None
    title_lower = title.lower()
    for make in known_makes:
        if make.lower() in title_lower:
            return make
    return None


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    # load processed.csv just to pull the cleaned `make` column
    processed_path = "data/processed.csv"
    raw_path       = "data/raw_listings.csv"

    print("Reading CSVs...")
    raw = pd.read_csv(raw_path)
    processed = pd.read_csv(processed_path) if os.path.exists(processed_path) else None

    # merge `make` from processed into raw (only if row counts match)
    if processed is not None and "make" in processed.columns and len(processed) == len(raw):
        raw["make"] = processed["make"].values
    else:
        # fallback: try to extract make from title using a known list
        known_makes = [
            "Toyota", "Honda", "Nissan", "Suzuki", "Mitsubishi", "Mazda",
            "Hyundai", "Kia", "BMW", "Mercedes", "Audi", "Volkswagen",
            "Perodua", "Isuzu", "Ford", "Jeep", "Land Rover",
        ]
        raw["make"] = raw["title"].apply(lambda t: extract_make(t, known_makes))

    print(f"Loaded {len(raw)} rows from raw_listings.csv")

    # ── build documents ────────────────────────────────────────────────────────

    now = datetime.now(timezone.utc)
    docs = []

    for _, row in raw.iterrows():
        price   = parse_price(row.get("price_raw"))
        year    = parse_year(row.get("year_raw"))
        mileage = parse_mileage(row.get("mileage_raw"))

        # skip rows with no price or url — they're useless for training
        if not price or not row.get("url"):
            continue

        doc = {
            "title":        str(row["title"]).strip() if not pd.isna(row.get("title")) else None,
            "make":         str(row["make"]).strip()  if not pd.isna(row.get("make"))  else None,
            "model":        None,          # not scraped yet — will be filled by scraper v2
            "variant":      None,
            "year":         year,
            "age":          (CURRENT_YEAR - year) if year else None,
            "mileage":      mileage,
            "price":        price,
            "fuel_type":    str(row["fuel_type"]).strip()    if not pd.isna(row.get("fuel_type"))    else None,
            "transmission": str(row["transmission"]).strip() if not pd.isna(row.get("transmission")) else None,
            "condition":    str(row["condition"]).strip()    if not pd.isna(row.get("condition"))     else None,
            "engine_cc":    parse_engine_cc(row.get("engine_cc")),
            "location":     str(row["location"]).strip()     if not pd.isna(row.get("location"))     else None,
            "source":       "riyasewana",
            "url":          str(row["url"]).strip(),
            "image_url":    None,
            "is_active":    True,
            "scraped_at":   now,
        }
        docs.append(doc)

    print(f"Built {len(docs)} valid documents (skipped {len(raw) - len(docs)} rows missing price/url)")

    # ── upsert into MongoDB ────────────────────────────────────────────────────

    client = MongoClient(MONGO_URI)
    col    = client[DB_NAME]["listings"]

    # ensure indexes exist
    col.create_index([("url", 1)], unique=True)
    col.create_index([("make", 1), ("model", 1), ("is_active", 1)])
    col.create_index([("scraped_at", -1)])
    print("Indexes ensured.")

    # bulk upsert — update if url exists, insert if not
    operations = [
        UpdateOne(
            {"url": doc["url"]},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        for doc in docs
    ]

    result = col.bulk_write(operations, ordered=False)
    print(f"\nDone!")
    print(f"  Inserted : {result.upserted_count}")
    print(f"  Updated  : {result.modified_count}")
    print(f"  Total    : {result.upserted_count + result.modified_count}")

    # quick sanity check
    total_in_db = col.count_documents({})
    print(f"  In DB now: {total_in_db} listings")

    client.close()


if __name__ == "__main__":
    main()
