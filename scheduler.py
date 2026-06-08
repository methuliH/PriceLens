"""
scheduler.py
------------
Runs both scrapers every 3 hours, marks stale listings, and retrains the
model when >= 30 new listings arrive.

Usage:
    python scheduler.py
"""

import importlib
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv
from pymongo import MongoClient
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────

os.makedirs("logs", exist_ok=True)

_fmt     = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_h  = RotatingFileHandler("logs/scheduler.log", maxBytes=5 * 1024 * 1024, backupCount=3)
_file_h.setFormatter(_fmt)
_cons_h  = logging.StreamHandler(sys.stdout)
_cons_h.setFormatter(_fmt)

log = logging.getLogger("scheduler")
log.setLevel(logging.INFO)
log.addHandler(_file_h)
log.addHandler(_cons_h)


# ── Config ────────────────────────────────────────────────────────────────────

MONGO_URI        = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME          = os.getenv("MONGO_DB", "price_prediction")
COLLECTION_NAME  = os.getenv("MONGO_COLLECTION", "listings")
RETRAIN_THRESHOLD = 30

SCRAPERS = [
    {"name": "riyasewana", "module": "scraper",        "pages": 50},
    {"name": "ikman",      "module": "ikman_scraper",   "pages": 50},
]


# ── Scheduled job ─────────────────────────────────────────────────────────────

def run_all():
    log.info("Starting scheduled run")
    total_new = 0

    # ── Run scrapers ──────────────────────────────────────────────────────────
    for scraper_cfg in SCRAPERS:
        name   = scraper_cfg["name"]
        module = scraper_cfg["module"]
        pages  = scraper_cfg["pages"]
        try:
            mod        = importlib.import_module(module)
            new_count  = mod.run(pages=pages)
            total_new += new_count
            log.info(f"{name}: {new_count} new listings")
        except Exception as exc:
            log.error(f"{name} scraper failed: {exc}", exc_info=True)

    # ── Mark stale listings ───────────────────────────────────────────────────
    try:
        client     = MongoClient(MONGO_URI)
        col        = client[DB_NAME][COLLECTION_NAME]
        cutoff     = datetime.now(timezone.utc) - timedelta(hours=24)
        stale_result = col.update_many(
            {"last_seen": {"$lt": cutoff}},
            {"$set": {"active": False}},
        )
        log.info(f"Marked {stale_result.modified_count} listings as stale (last_seen > 24h)")
        client.close()
    except Exception as exc:
        log.error(f"Stale listing cleanup failed: {exc}", exc_info=True)

    # ── Conditional retrain ───────────────────────────────────────────────────
    if total_new >= RETRAIN_THRESHOLD:
        log.info(f"New listings ({total_new}) >= threshold ({RETRAIN_THRESHOLD}) — retraining")
        try:
            from ml.train import train
            metrics = train()
            log.info(
                f"Retrain complete — MAE: {metrics.get('MAE', 'n/a'):,.0f} | "
                f"R2: {metrics.get('R2', 'n/a'):.4f} | "
                f"MAPE: {metrics.get('MAPE (%)', 'n/a'):.2f}%"
            )
        except Exception as exc:
            log.error(f"Retrain failed: {exc}", exc_info=True)
    else:
        log.info(
            f"Skipping retrain — only {total_new} new listings (threshold: {RETRAIN_THRESHOLD})"
        )

    log.info("Scheduled run complete")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="Asia/Colombo")
    scheduler.add_job(
        run_all,
        trigger=IntervalTrigger(hours=3),
        next_run_time=datetime.now(timezone.utc),
    )
    log.info("Scheduler started — running every 3 hours (Asia/Colombo). Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")
