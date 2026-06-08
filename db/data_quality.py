"""
db/data_quality.py
------------------
One-shot script: applies location normalisation from ml/preprocess.py
to every listing document in MongoDB that currently has a non-canonical
location value.

Run once after deploying the improve/data-quality branch:
    python db/data_quality.py

Safe to re-run — only updates documents where location actually changes.
"""

import os
import sys
import logging

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.preprocess import normalize_location  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BATCH_SIZE = 500


def run():
    load_dotenv()
    uri = os.getenv("MONGO_URI")
    if not uri:
        raise RuntimeError("MONGO_URI not set")
    db_name = os.getenv("MONGO_DB", "price_prediction")

    client = MongoClient(uri)
    col = client[db_name]["listings"]

    docs = list(col.find({}, {"_id": 1, "location": 1}))
    log.info(f"Loaded {len(docs)} documents")

    ops = []
    changed = 0
    for doc in docs:
        raw = doc.get("location")
        normalised = normalize_location(raw)
        if normalised != raw:
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"location": normalised}}))
            changed += 1

    if not ops:
        log.info("All locations already normalised — nothing to update")
        client.close()
        return

    log.info(f"Updating {changed} documents in batches of {BATCH_SIZE}...")
    total_modified = 0
    for i in range(0, len(ops), BATCH_SIZE):
        result = col.bulk_write(ops[i:i + BATCH_SIZE], ordered=False)
        total_modified += result.modified_count

    client.close()
    log.info(f"Done. {total_modified} documents updated.")


if __name__ == "__main__":
    run()
