"""
extract_models.py — backfill the `model` field on existing MongoDB listings
by parsing the model name from the URL slug, with a title-based fallback.

Run once: python db/extract_models.py
"""

import os
import re
import logging
from pymongo import MongoClient, UpdateOne
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME", "pricelens")

RIYASEWANA_PREFIX = "riyasewana.com/buy/"
JUNK_WORDS        = {"car", "vehicle", "used", "sale"}
BATCH_SIZE        = 200


# ── extraction helpers ────────────────────────────────────────────────────────

def _slug_from_url(url: str) -> str | None:
    """Return the path slug after /buy/, or None if not a riyasewana URL."""
    idx = url.find(RIYASEWANA_PREFIX)
    if idx == -1:
        return None
    return url[idx + len(RIYASEWANA_PREFIX):].strip("/")


def extract_model_from_url(url: str | None, make: str | None) -> str | None:
    if not url:
        return None

    slug = _slug_from_url(url)
    if not slug or "-sale-" not in slug:
        return None

    # strip make prefix from slug (first occurrence only)
    if make:
        make_slug = make.lower().replace(" ", "-")
        prefix = make_slug + "-"
        if slug.startswith(prefix):
            slug = slug[len(prefix):]

    # drop everything from -sale- onward
    model_slug = slug.split("-sale-")[0]
    if not model_slug:
        return None

    return model_slug.replace("-", " ").title()


def extract_model_from_title(title: str | None, make: str | None) -> str | None:
    if not title or not make:
        return None

    words = title.split()
    make_words = make.split()
    n = len(make_words)

    # find where the make ends in the title (case-insensitive)
    start_idx = 0
    for i in range(len(words) - n + 1):
        if [w.lower() for w in words[i : i + n]] == [w.lower() for w in make_words]:
            start_idx = i + n
            break

    if start_idx >= len(words):
        return None

    # take the next token, strip punctuation
    candidate = words[start_idx].strip("(),.;:")
    return candidate.title() if candidate else None


def _is_junk(model: str | None) -> bool:
    if not model:
        return True
    if re.fullmatch(r"\d{4}", model.strip()):
        return True
    if model.strip().lower() in JUNK_WORDS:
        return True
    return False


def resolve_model(url: str | None, title: str | None, make: str | None) -> tuple[str | None, str]:
    """Return (model, method) where method is 'url', 'title', or 'failed'."""
    candidate = extract_model_from_url(url, make)
    if not _is_junk(candidate):
        return candidate, "url"

    candidate = extract_model_from_title(title, make)
    if not _is_junk(candidate):
        return candidate, "title"

    return None, "failed"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    client = MongoClient(MONGO_URI)
    col    = client[DB_NAME]["listings"]

    query = {"$or": [{"model": None}, {"model": {"$exists": False}}]}
    total = col.count_documents(query)
    print(f"Documents with model=null: {total}\n")

    cursor = col.find(query, {"_id": 1, "url": 1, "title": 1, "make": 1})

    processed    = 0
    url_count    = 0
    title_count  = 0
    failed_count = 0
    batch: list[UpdateOne] = []

    for doc in cursor:
        model, method = resolve_model(
            doc.get("url"),
            doc.get("title"),
            doc.get("make"),
        )

        if method == "url":
            url_count += 1
        elif method == "title":
            title_count += 1
        else:
            failed_count += 1
            log.warning(
                "Could not extract model — _id=%s url=%s title=%s",
                doc["_id"], doc.get("url"), doc.get("title"),
            )

        if model is not None:
            batch.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"model": model}}))

        processed += 1

        if processed % BATCH_SIZE == 0:
            if batch:
                col.bulk_write(batch, ordered=False)
                batch.clear()
            print(f"Processed {processed}/{total}...")

    # flush remainder
    if batch:
        col.bulk_write(batch, ordered=False)

    # final progress line if last batch wasn't a clean multiple
    if total == 0 or processed % BATCH_SIZE != 0:
        print(f"Processed {processed}/{total}...")

    print()
    print(f"  Total processed        : {processed}")
    print(f"  Successfully extracted : {url_count + title_count}")
    print(f"  Fell back to title parse : {title_count}")
    print(f"  Failed (left as null)  : {failed_count}")

    client.close()


if __name__ == "__main__":
    main()
