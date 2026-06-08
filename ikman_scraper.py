"""
ikman_scraper.py
----------------
Scrapes vehicle listings from ikman.lk and upserts them to MongoDB.

Card data extracted from SSR HTML (stable data-testid attributes).
Detail data extracted from embedded JSON in each listing page.

Usage:
    python ikman_scraper.py --pages 20
"""

import os
import re
import time
import random
import logging
import argparse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME         = os.getenv("MONGO_DB", "price_prediction")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "listings")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language":        "en-US,en;q=0.9",
    "Accept-Encoding":        "gzip, deflate, br",
    "Connection":             "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest":         "document",
    "Sec-Fetch-Mode":         "navigate",
    "Sec-Fetch-Site":         "none",
}

BASE_URL = "https://ikman.lk/en/ads/sri-lanka/cars"

# Maps embedded JSON attribute keys → our field names
_ATTR_KEY_MAP = {
    "brand":           "make",
    "model":           "model",
    "model_year":      "year",
    "condition":       "condition",
    "transmission":    "transmission",
    "fuel_type":       "fuel_type",
    "engine_capacity": "engine_cc_raw",
    "mileage":         "mileage_raw",
}


# ── Fetch helper ──────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    log.error(f"All retries failed for {url}")
    return None


# ── Card parser (SERP page) ───────────────────────────────────────────────────

def parse_listing_card(card) -> dict:
    """
    Extract fields from a listing <li> element on the search results page.
    Uses data-testid='ad-card-link' anchor (stable across deployments) and
    regex on the rendered anchor text for mileage / price.
    """
    a = card.find("a", attrs={"data-testid": "ad-card-link"})
    if not a:
        return {}

    href  = a.get("href", "")
    url   = f"https://ikman.lk{href}" if href.startswith("/") else href
    title = (a.get("title") or "").strip()
    if not title:
        h2 = a.find("h2")
        title = h2.get_text(strip=True) if h2 else ""

    text = a.get_text(separator=" ", strip=True)

    price_m   = re.search(r"Rs\s*([\d,]+)", text)
    mileage_m = re.search(r"([\d,]+)\s*km", text)
    year_m    = re.search(r"\b(19|20)\d{2}\b", title)

    # Location appears as "CityName, Cars" in the anchor text
    loc_m = re.search(r"([A-Za-z][A-Za-z\s]+),\s*Cars", text)

    return {
        "url":      url,
        "title":    title,
        "price":    float(price_m.group(1).replace(",", ""))   if price_m   else None,
        "mileage":  float(mileage_m.group(1).replace(",", "")) if mileage_m else None,
        "year":     int(year_m.group())                        if year_m    else None,
        "location": loc_m.group(1).strip()                     if loc_m     else None,
    }


# ── Detail parser (individual listing page) ──────────────────────────────────

def parse_listing_detail(url: str) -> dict:
    """
    Fetch the listing page and extract make/model/fuel/transmission/condition
    from the embedded Redux state JSON (\"ad\":{...} block).
    """
    resp = _get(url)
    if not resp:
        return {}

    raw = resp.text
    ad_start = raw.find('"ad":{')
    if ad_start == -1:
        return {}

    snippet = raw[ad_start: ad_start + 5000]

    # Parse attributes array: {"label":"Brand","value":"Toyota","key":"brand",...}
    raw_attrs = re.findall(
        r'"label":"([^"]+)","value":"([^"]+)","key":"([^"]+)"',
        snippet,
    )
    attr_by_key = {key: value for _, value, key in raw_attrs}

    data: dict = {}
    for json_key, field in _ATTR_KEY_MAP.items():
        if json_key in attr_by_key:
            data[field] = attr_by_key[json_key]

    # Clean numeric fields
    if "mileage_raw" in data:
        m = re.search(r"[\d,]+", data.pop("mileage_raw"))
        data["mileage"] = float(m.group().replace(",", "")) if m else None

    if "engine_cc_raw" in data:
        m = re.search(r"[\d,]+", data.pop("engine_cc_raw"))
        data["engine_cc"] = int(m.group().replace(",", "")) if m else None

    if "year" in data:
        y = data["year"]
        data["year"] = int(y) if str(y).isdigit() else None

    # Location from JSON (more precise city name than card text)
    loc_m = re.search(r'"location":\{"id":\d+,"name":"([^"]+)"', snippet)
    if loc_m:
        data["location"] = loc_m.group(1)

    # Price from JSON (fallback if card parse missed it)
    price_m = re.search(r'"amount":"Rs\s*([\d,]+)"', snippet)
    if price_m:
        data["price"] = float(price_m.group(1).replace(",", ""))

    return data


# ── Scrape loop ───────────────────────────────────────────────────────────────

def scrape(num_pages: int = 10, detail_pages: bool = True) -> list[dict]:
    all_records: list[dict] = []

    for page_num in range(1, num_pages + 1):
        url = BASE_URL if page_num == 1 else f"{BASE_URL}?page={page_num}"
        log.info(f"Scraping page {page_num}/{num_pages} -> {url}")

        resp = _get(url)
        if not resp:
            continue

        soup  = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("li.gtm-normal-ad, li.gtm-top-ad")

        if not cards:
            log.warning(f"No listing cards on page {page_num} — stopping pagination")
            break

        log.info(f"  Found {len(cards)} listings on page {page_num}")

        for card in cards:
            record = parse_listing_card(card)
            if not record.get("url"):
                continue

            if detail_pages:
                time.sleep(random.uniform(1.5, 3.0))
                detail = parse_listing_detail(record["url"])
                # Detail fields win over card fields (more precise)
                record.update(detail)

            all_records.append(record)

        time.sleep(random.uniform(2.0, 4.0))

    log.info(f"Scraping complete. Total records: {len(all_records)}")
    return all_records


# ── MongoDB ───────────────────────────────────────────────────────────────────

def save_to_mongo(records: list[dict]) -> int:
    client = MongoClient(MONGO_URI)
    col    = client[DB_NAME][COLLECTION_NAME]

    now = datetime.now(timezone.utc)
    ops = []
    for record in records:
        r = dict(record)
        r["source"]    = "ikman"
        r["last_seen"] = now
        r["active"]    = True
        if not r.get("url"):
            continue
        ops.append(UpdateOne({"url": r["url"]}, {"$set": r}, upsert=True))

    if not ops:
        client.close()
        return 0

    result = col.bulk_write(ops, ordered=False)
    client.close()
    log.info(f"ikman: {result.upserted_count} new / {result.modified_count} updated")
    return result.upserted_count


def run(pages: int = 50) -> int:
    records = scrape(num_pages=pages, detail_pages=True)
    return save_to_mongo(records)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape ikman.lk vehicle listings")
    parser.add_argument("--pages",      type=int, default=20)
    parser.add_argument("--no-details", action="store_true")
    args = parser.parse_args()

    records = scrape(num_pages=args.pages, detail_pages=not args.no_details)
    new_inserts = save_to_mongo(records)
    print(f"Done: {len(records)} scraped, {new_inserts} new inserts")


if __name__ == "__main__":
    main()
