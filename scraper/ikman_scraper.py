"""
ikman_scraper.py
----------------
Scrapes vehicle listings from ikman.lk and upserts them directly into MongoDB.
All fields are extracted in a single pass (listing card + detail page).

Usage:
    python scraper/ikman_scraper.py --pages 20
    python scraper/ikman_scraper.py --pages 5 --no-details
"""

import argparse
import logging
import os
import random
import re
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient

from scraper.utils import get_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_URL     = "https://ikman.lk/en/ads/sri-lanka/vehicles"
IKMAN_ROOT   = "https://ikman.lk"
CURRENT_YEAR = 2026


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_price(raw: str) -> float | None:
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[^\d]", "", raw)
    try:
        return float(cleaned) or None
    except ValueError:
        return None


def _parse_mileage(raw: str) -> float | None:
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[^\d]", "", raw)
    try:
        val = float(cleaned)
        return val if val < 2_000_000 else None
    except ValueError:
        return None


def _parse_engine_cc(raw: str) -> int | None:
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[^\d]", "", raw)
    return int(cleaned) if cleaned else None


def _parse_year(raw: str) -> int | None:
    if not isinstance(raw, str):
        return None
    m = re.search(r"\b(19[789]\d|20[012]\d)\b", raw)
    return int(m.group()) if m else None


# ── Card parsing ──────────────────────────────────────────────────────────────

def parse_ikman_card(li) -> dict:
    """Extract fields from a single listing card on the search results page."""
    data: dict = {}
    try:
        a_tag = li.find("a", href=True)
        if not a_tag:
            return data

        href = a_tag.get("href", "")
        data["url"] = (IKMAN_ROOT + href) if href.startswith("/") else href

        h3 = a_tag.find("h3")
        if h3:
            data["title"] = h3.get_text(strip=True)

        img = a_tag.find("img")
        if img:
            data["image_url"] = img.get("src")

        skip_words = ("member", "urgent", "top ad")
        for p in a_tag.find_all("p"):
            text = p.get_text(strip=True)
            if not text or text.lower() in skip_words:
                continue
            tl = text.lower()
            if "rs" in tl and "price_raw" not in data:
                data["price_raw"] = text
            elif "km" in tl and "mileage_raw" not in data:
                data["mileage_raw"] = text
            elif "," in text and "location" not in data and "rs" not in tl and "km" not in tl:
                data["location"] = text.split(",")[0].strip()

    except Exception as e:
        log.debug(f"Error parsing card: {e}")

    return data


# ── Detail page parsing ───────────────────────────────────────────────────────

_DL_FIELD_MAP = {
    "brand":               "make",
    "model":               "model",
    "trim / edition":      "variant",
    "trim/edition":        "variant",
    "year of manufacture": "year_raw",
    "condition":           "condition",
    "transmission":        "transmission",
    "fuel type":           "fuel_type",
    "engine capacity":     "engine_cc_raw",
    "mileage":             "mileage_raw",
}


def fetch_ikman_detail(url: str) -> dict:
    """Extract vehicle specs from a listing's detail page via the <dl> spec table."""
    if not url:
        return {}

    soup = get_page(url)
    if not soup:
        return {}

    detail: dict = {}
    try:
        for dl in soup.find_all("dl"):
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                key   = dt.get_text(strip=True).rstrip(":").strip().lower()
                value = dd.get_text(strip=True)
                field = _DL_FIELD_MAP.get(key)
                if field and value:
                    detail[field] = value
    except Exception as e:
        log.debug(f"Error parsing detail page {url}: {e}")

    return detail


# ── Document builder ──────────────────────────────────────────────────────────

def build_document(record: dict, now: datetime) -> dict | None:
    url = record.get("url")
    if not url:
        return None

    price   = _parse_price(record.get("price_raw"))
    mileage = _parse_mileage(record.get("mileage_raw"))
    year    = _parse_year(str(record.get("year_raw", "")))
    engine  = _parse_engine_cc(record.get("engine_cc_raw", ""))

    return {
        "title":        record.get("title"),
        "make":         record.get("make"),
        "model":        record.get("model"),
        "variant":      record.get("variant"),
        "year":         year,
        "age":          (CURRENT_YEAR - year) if year else None,
        "mileage":      mileage,
        "price":        price,
        "fuel_type":    record.get("fuel_type"),
        "transmission": record.get("transmission"),
        "condition":    record.get("condition"),
        "engine_cc":    engine,
        "location":     record.get("location"),
        "source":       "ikman",
        "url":          url,
        "image_url":    record.get("image_url"),
        "is_active":    True,
        "scraped_at":   now,
    }


# ── Main scrape loop ──────────────────────────────────────────────────────────

def scrape(col, num_pages: int, detail_pages: bool, delay: float) -> None:
    total = 0

    for page_num in range(1, num_pages + 1):
        url  = f"{BASE_URL}?page={page_num}"
        log.info(f"Scraping page {page_num}/{num_pages} → {url}")

        soup = get_page(url)
        if not soup:
            log.warning(f"Failed to fetch page {page_num}, skipping.")
            continue

        cards = [li for li in soup.select("li") if li.find("a", href=True) and li.find("h3")]
        if not cards:
            log.warning(f"No listing cards found on page {page_num} — stopping.")
            break

        log.info(f"  Found {len(cards)} listings")
        now = datetime.now(timezone.utc)

        for card in cards:
            record = parse_ikman_card(card)
            if not record.get("url"):
                continue

            if detail_pages:
                time.sleep(delay + random.uniform(0, 1.5))
                record.update(fetch_ikman_detail(record["url"]))

            doc = build_document(record, now)
            if doc and doc.get("price"):
                col.update_one(
                    {"url": doc["url"]},
                    {"$set": doc, "$setOnInsert": {"created_at": now}},
                    upsert=True,
                )
                total += 1

        log.info(f"  Running total: {total} upserted")
        time.sleep(random.uniform(3.0, 5.0))

    log.info(f"Done! Total listings upserted: {total}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scrape ikman.lk vehicle listings into MongoDB")
    parser.add_argument("--pages",      type=int,   default=20,  help="Search result pages to scrape")
    parser.add_argument("--delay",      type=float, default=3.0, help="Base delay (s) between detail requests")
    parser.add_argument("--no-details", action="store_true",     help="Skip detail pages (faster, fewer fields)")
    args = parser.parse_args()

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set — add it to .env")
    db_name = os.getenv("DB_NAME", "pricelens")

    client = MongoClient(uri)
    col    = client[db_name]["listings"]

    col.create_index([("url", 1)], unique=True)
    col.create_index([("make", 1), ("model", 1), ("is_active", 1)])
    col.create_index([("scraped_at", -1)])
    log.info(f"Connected → {db_name}.listings")

    try:
        scrape(col, num_pages=args.pages, detail_pages=not args.no_details, delay=args.delay)
    finally:
        client.close()


if __name__ == "__main__":
    main()
