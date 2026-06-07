"""
ikman_scraper.py
----------------
Scrapes vehicle listings from ikman.lk and upserts them to MongoDB.

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

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGO_DB", "price_prediction")
COLLECTION_NAME  = os.getenv("MONGO_COLLECTION", "listings")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://ikman.lk/en/ads/sri-lanka/cars"


# ── Fetch helpers ─────────────────────────────────────────────────────────────

def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)
    log.error(f"All retries failed for {url}")
    return None


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_price(raw: str | None) -> float | None:
    if not raw:
        return None
    digits = re.sub(r"[^\d]", "", raw)
    return float(digits) if digits else None


def parse_listing_card(card) -> dict:
    data: dict = {}
    try:
        link = card.select_one("a[href]")
        if link:
            href = link.get("href", "")
            data["url"] = f"https://ikman.lk{href}" if href.startswith("/") else href

        title_tag = card.select_one("h2, .title, [class*='title']")
        if title_tag:
            data["title"] = title_tag.get_text(strip=True)

        price_tag = card.select_one("[class*='price']")
        data["price"] = parse_price(price_tag.get_text(strip=True) if price_tag else None)

        location_tag = card.select_one("[class*='location'], [class*='area']")
        if location_tag:
            data["location"] = location_tag.get_text(strip=True)

        year_match = re.search(r"\b(19|20)\d{2}\b", data.get("title", ""))
        if year_match:
            data["year"] = int(year_match.group())

    except Exception as e:
        log.debug(f"Error parsing card: {e}")

    return data


def parse_listing_detail(url: str) -> dict:
    if not url:
        return {}
    soup = get_page(url)
    if not soup:
        return {}

    data: dict = {}
    try:
        for row in soup.select("table tr, .details li, [class*='detail'] li"):
            cells = row.find_all(["td", "th", "span"])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower().replace(" ", "_")
                val = cells[1].get_text(strip=True)
                data[key] = val

        for key_hint, field in [("make", "make"), ("brand", "make"), ("model", "model"),
                                 ("mileage", "mileage"), ("fuel", "fuel_type"),
                                 ("transmission", "transmission"), ("condition", "condition")]:
            for tag in soup.select(f"[data-key='{key_hint}'], [class*='{key_hint}']"):
                if field not in data:
                    data[field] = tag.get_text(strip=True)

        mileage_tag = soup.select_one("[class*='mileage'], [data-key='mileage']")
        if mileage_tag and "mileage" not in data:
            raw = mileage_tag.get_text(strip=True)
            digits = re.sub(r"[^\d]", "", raw)
            data["mileage"] = float(digits) if digits else None

    except Exception as e:
        log.debug(f"Error parsing detail {url}: {e}")

    return data


# ── Scrape loop ───────────────────────────────────────────────────────────────

def scrape(num_pages: int = 10, detail_pages: bool = True) -> list[dict]:
    all_records: list[dict] = []

    for page_num in range(1, num_pages + 1):
        url = f"{BASE_URL}?sort_by=date&order=desc&page={page_num}"
        log.info(f"Scraping page {page_num}/{num_pages} -> {url}")

        soup = get_page(url)
        if not soup:
            continue

        cards = soup.select("li.listing, [class*='item'], [class*='ad-listing']")
        if not cards:
            log.warning(f"No listing cards on page {page_num} — stopping")
            break

        log.info(f"  Found {len(cards)} listings on page {page_num}")

        for card in cards:
            record = parse_listing_card(card)
            if not record.get("url"):
                continue
            if detail_pages:
                time.sleep(random.uniform(1.5, 3.0))
                record.update(parse_listing_detail(record["url"]))
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
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--no-details", action="store_true")
    args = parser.parse_args()

    records = scrape(num_pages=args.pages, detail_pages=not args.no_details)
    new_inserts = save_to_mongo(records)
    print(f"Done: {len(records)} scraped, {new_inserts} new inserts")


if __name__ == "__main__":
    main()
