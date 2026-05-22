"""
scrape_details.py
-----------------
Fetches detail pages for listings that are missing fuel/transmission/condition.
Runs in small batches with long pauses to avoid 429 rate limiting.
Saves progress after every batch — safe to stop and resume anytime.

Usage:
    python scrape_details.py
    python scrape_details.py --input data/raw_listings.csv --batch 10 --pause 45
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                log.warning(f"429 rate limit hit. Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def fetch_detail(url: str) -> dict:
    """Extract fuel type, transmission, condition from a listing detail page."""
    if not isinstance(url, str) or not url.startswith("http"):
        return {}

    soup = get_page(url)
    if not soup:
        return {}

    detail = {}
    try:
        # Data is in span.qspec elements inside div.quick-specs
        # e.g: "2013", "Hybrid", "Automatic", "118,300 km", "1330cc"
        qspecs = soup.select("div.quick-specs span.qspec")
        for span in qspecs:
            text = span.get_text(strip=True)
            tl   = text.lower()

            if any(f in tl for f in ["petrol", "diesel", "hybrid", "electric", "gas"]):
                detail["fuel_type"] = text
            elif any(t in tl for t in ["automatic", "manual", "cvt", "tiptronic"]):
                detail["transmission"] = text
            elif "cc" in tl:
                detail["engine_cc"] = text
            elif "km" in tl:
                pass  # already have mileage from card

        # Also try detail-row divs for condition
        for row in soup.select("div.detail-card div.detail-row"):
            text = row.get_text(separator=" ", strip=True).lower()
            if "condition" in text:
                val = row.get_text(separator=" ", strip=True)
                detail["condition"] = val.replace("Condition", "").strip()

    except Exception as e:
        log.debug(f"Parse error for {url}: {e}")

    return detail


def needs_details(row) -> bool:
    """Return True if this row is still missing detail data."""
    missing = ["fuel_type", "transmission", "condition"]
    for col in missing:
        if col not in row.index:
            return True
        val = str(row[col]) if not pd.isna(row[col]) else ""
        if val in ("", "nan", "Unknown", "Other"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw_listings.csv")
    parser.add_argument("--batch",  type=int, default=10,  help="Listings per batch")
    parser.add_argument("--pause",  type=int, default=45,  help="Seconds to pause between batches")
    parser.add_argument("--delay",  type=float, default=4.0, help="Seconds between individual requests")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    log.info(f"Loaded {len(df)} rows from {args.input}")

    # Identify rows that still need details
    todo_mask = df.apply(needs_details, axis=1)
    todo_indices = df[todo_mask].index.tolist()

    log.info(f"Rows needing details: {len(todo_indices)} / {len(df)}")

    if not todo_indices:
        log.info("All rows already have details. Nothing to do.")
        return

    # Process in batches
    total_batches = (len(todo_indices) + args.batch - 1) // args.batch

    for batch_num, batch_start in enumerate(range(0, len(todo_indices), args.batch), start=1):
        batch_indices = todo_indices[batch_start: batch_start + args.batch]
        log.info(f"── Batch {batch_num}/{total_batches} ({len(batch_indices)} listings) ──")

        for i, idx in enumerate(batch_indices):
            url = df.at[idx, "url"]
            log.info(f"  [{i+1}/{len(batch_indices)}] {url}")

            detail = fetch_detail(url)

            if detail:
                for key, val in detail.items():
                    df.at[idx, key] = val
                log.info(f"    → Got: {detail}")
            else:
                log.info(f"    → No detail data found")

            # Delay between individual requests
            time.sleep(args.delay + random.uniform(0, 2))

        # Save after every batch
        df.to_csv(args.input, index=False)
        log.info(f"Progress saved → {args.input}")

        # Longer pause between batches (skip pause after last batch)
        if batch_num < total_batches:
            log.info(f"Pausing {args.pause}s before next batch...")
            time.sleep(args.pause)

    log.info("Done! All details fetched.")

    # Summary
    for col in ["fuel_type", "transmission", "condition"]:
        if col in df.columns:
            filled = df[col].notna().sum()
            print(f"{col}: {filled}/{len(df)} filled")


if __name__ == "__main__":
    main()