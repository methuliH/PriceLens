"""
scraper.py
----------
Scrapes vehicle listings from riyasewana.com and saves raw data to CSV.

Usage:
    python scraper.py --pages 20 --output data/raw_listings.csv
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import argparse
import os
import logging
from datetime import datetime, timezone

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

BASE_URL = "https://riyasewana.com/search/cars"


def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a page and return a BeautifulSoup object."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            time.sleep(2 ** attempt)  # exponential backoff
    log.error(f"All retries failed for {url}")
    return None


def parse_listing_card(card) -> dict:
    """
    Extract fields from a single listing card on the search results page.
    Based on riyasewana.com's actual HTML structure (li.v-card).
    """
    data = {}

    try:
        # Title + URL — from the anchor with a title attribute
        link_tag = card.select_one("a[title]")
        if link_tag:
            data["title"] = link_tag.get("title", "").strip()
            data["url"]   = link_tag.get("href", "").strip()

        # Price — div.v-card-price → "Rs. 6,690,000"
        price_tag = card.select_one(".v-card-price")
        data["price_raw"] = price_tag.get_text(strip=True) if price_tag else None

        # Year — div.v-card-year → "2020"
        year_tag = card.select_one(".v-card-year")
        data["year_raw"] = year_tag.get_text(strip=True) if year_tag else None

        # Meta — div.v-card-meta contains location · mileage
        # Text looks like: "Kandana · 146,000 km"  (SVG icons are in between)
        meta_tag = card.select_one(".v-card-meta")
        if meta_tag:
            # Remove SVG elements so we only get text
            for svg in meta_tag.find_all("svg"):
                svg.decompose()
            meta_text = meta_tag.get_text(separator=" ", strip=True)
            # Split on the · separator
            parts = [p.strip() for p in meta_text.split("·")]
            if len(parts) >= 1:
                data["location"]    = parts[0]
            if len(parts) >= 2:
                data["mileage_raw"] = parts[1]

    except Exception as e:
        log.debug(f"Error parsing card: {e}")

    return data


def parse_listing_detail(url: str) -> dict:
    """
    Fetch the individual listing page for richer details.
    This catches year, make, model separately if not in the card.
    """
    if not url:
        return {}

    # Make absolute URL if relative
    if url.startswith("/"):
        url = "https://riyasewana.com" + url

    soup = get_page(url)
    if not soup:
        return {}

    data = {}
    try:
        # Detail table / info list on the listing page
        rows = soup.select("table.moretbl tr, .listing-info tr, .detail-table tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower().replace(" ", "_")
                val = cells[1].get_text(strip=True)
                data[key] = val

        # Sometimes year is in the detail fields
        year_tag = soup.select_one(".year, [data-field='year']")
        if year_tag:
            data["year"] = year_tag.get_text(strip=True)

    except Exception as e:
        log.debug(f"Error parsing detail page {url}: {e}")

    return data


def scrape(num_pages: int = 10, detail_pages: bool = True) -> pd.DataFrame:
    """
    Main scraping loop. Iterates over search result pages.

    Args:
        num_pages:    How many search result pages to scrape.
        detail_pages: Whether to fetch each listing's detail page (slower but richer).
    """
    all_records = []

    for page_num in range(1, num_pages + 1):
        url = f"{BASE_URL}?page={page_num}" if page_num > 1 else BASE_URL
        log.info(f"Scraping page {page_num}/{num_pages} → {url}")

        soup = get_page(url)
        if not soup:
            continue

        # Find listing cards
        cards = soup.select("li.v-card")

        if not cards:
            log.warning(f"No listing cards found on page {page_num}. Check your CSS selectors.")
            break

        log.info(f"  Found {len(cards)} listings on page {page_num}")

        for card in cards:
            record = parse_listing_card(card)

            if detail_pages and record.get("url"):
                time.sleep(random.uniform(1.5, 3.0))  # be polite
                detail = parse_listing_detail(record["url"])
                record.update(detail)

            all_records.append(record)

        # Polite delay between pages
        time.sleep(random.uniform(2.0, 4.0))

    df = pd.DataFrame(all_records)
    log.info(f"Scraping complete. Total records: {len(df)}")
    return df


def save_to_mongo(records: list[dict]) -> int:
    load_dotenv()
    uri             = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name         = os.getenv("MONGO_DB", "price_prediction")
    collection_name = os.getenv("MONGO_COLLECTION", "listings")

    client = MongoClient(uri)
    col    = client[db_name][collection_name]

    now = datetime.now(timezone.utc)
    ops = []
    for record in records:
        r = dict(record)
        r["source"]    = "riyasewana"
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
    log.info(f"riyasewana: {result.upserted_count} new / {result.modified_count} updated")
    return result.upserted_count


def run(pages: int = 50) -> int:
    df = scrape(num_pages=pages, detail_pages=True)
    new_inserts = save_to_mongo(df.to_dict("records"))
    return new_inserts


def main():
    parser = argparse.ArgumentParser(description="Scrape riyasewana.com vehicle listings")
    parser.add_argument("--pages", type=int, default=20, help="Number of search result pages to scrape")
    parser.add_argument("--output", type=str, default="data/raw_listings.csv", help="Output CSV path")
    parser.add_argument("--no-details", action="store_true", help="Skip fetching individual listing pages (faster)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = scrape(num_pages=args.pages, detail_pages=not args.no_details)

    df.to_csv(args.output, index=False)
    log.info(f"Saved {len(df)} records to {args.output}")
    print(df.head())


if __name__ == "__main__":
    main()