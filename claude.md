# PriceLens — Claude Code Context

## Project overview
A full-stack Sri Lankan used vehicle price predictor. Scrapes riyasewana.com, trains an XGBoost model, and serves predictions via FastAPI + a vanilla HTML/CSS/JS dashboard. All listing data is persisted in MongoDB.

## Stack
- **Scraping**: requests + BeautifulSoup
- **ML**: pandas, scikit-learn, XGBoost, SHAP, joblib
- **Backend**: FastAPI + uvicorn
- **Database**: MongoDB via Motor async driver (`db/mongo.py`)
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
- **Python**: 3.13 on Windows

## Project structure
```
PriceLens/
├── api.py                       # FastAPI backend (root for easy uvicorn)
├── requirements.txt
├── Makefile
│
├── scraper/
│   ├── scraper.py               # scrapes riyasewana.com search pages
│   └── scrape_details.py        # fills in fuel/transmission/condition from detail pages
│
├── ml/
│   ├── preprocess.py            # cleans raw CSV, engineers features
│   ├── train.py                 # trains XGBoost, outputs model + SHAP plots
│   └── predict.py               # CLI prediction tool
│
├── frontend/
│   └── index.html               # dashboard UI
│
├── db/
│   ├── mongo.py                 # Motor async client, lazy singleton
│   └── migrate.py               # one-shot CSV → MongoDB migration script
│
├── data/
│   ├── raw_listings.csv         # scraped data (gitignored)
│   └── processed.csv            # cleaned, encoded, model-ready (gitignored)
├── models/
│   └── price_model.joblib       # saved bundle (gitignored)
└── outputs/plots/               # actual_vs_predicted.png, feature_importance.png, shap_summary.png
```

## MongoDB architecture
- Database: `pricelens` (override with `DB_NAME` env var)
- Collections:
  - `listings` — vehicle listings from riyasewana.com, upserted by URL
  - `scrape_runs` — scraper run metadata (reserved for future use)
- `db/mongo.py`: lazy singleton `AsyncIOMotorClient`, reads `MONGO_URI` and `DB_NAME` from env
- `db/migrate.py`: one-shot bulk-upsert from `data/raw_listings.csv` into MongoDB

### Listing document schema
```python
{
  "title":        str,
  "make":         str | None,
  "model":        str | None,       # filled by scraper v2
  "variant":      str | None,
  "year":         int | None,
  "age":          int | None,
  "mileage":      float | None,
  "price":        float | None,
  "fuel_type":    str | None,
  "transmission": str | None,
  "condition":    str | None,
  "engine_cc":    int | None,
  "location":     str | None,
  "source":       "riyasewana",
  "url":          str,              # unique index
  "image_url":    str | None,
  "is_active":    bool,
  "scraped_at":   datetime,
  "created_at":   datetime,
}
```

## Environment variables
| Variable   | Default                      | Description                    |
|------------|------------------------------|--------------------------------|
| `MONGO_URI`| *(required)*                 | MongoDB connection string      |
| `DB_NAME`  | `pricelens`                  | MongoDB database name          |

## Model bundle
`joblib.load("models/price_model.joblib")` returns a dict:
```python
{
  "model":          xgb.XGBRegressor,
  "label_encoders": {"make": LabelEncoder, "location": LabelEncoder},
  "feature_cols":   [...],   # ordered list of feature names the model expects
  "metrics":        {"MAE": ..., "RMSE": ..., "R²": ..., "MAPE (%)": ...},
  "log_transform":  True,    # predictions must be back-transformed with np.exp()
}
```

## Feature engineering (must match ml/preprocess.py exactly)
- `age = 2026 - year`
- `log_mileage = np.log1p(mileage)`
- `make`, `location` → LabelEncoded (use saved encoders; unseen values → -1)
- `fuel_type`, `transmission`, `condition` → one-hot encoded with prefix (e.g. `fuel_type_Hybrid`)

## Key conventions
- Prices are in LKR (Sri Lankan Rupees)
- CURRENT_YEAR = 2026
- Unseen categorical values → encode as -1 (not error)
- Confidence range = predicted_price ± 12%

## API (api.py)
- FastAPI app with CORS enabled (allow all origins for dev)
- Model and `.env` loaded at startup via lifespan context manager + python-dotenv
- All data endpoints read live from MongoDB; no CSV reads at runtime

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check; reports `model_loaded` status |
| `POST` | `/predict` | Predict price → returns prediction + ±12% range + up to 5 comparable MongoDB listings |
| `GET` | `/makes` | Distinct active makes from MongoDB |
| `GET` | `/models/{make}` | Distinct active models for a given make from MongoDB |
| `GET` | `/market-stats` | Aggregate price stats + top 5 makes + fuel distribution |
| `GET` | `/listings` | Paginated active listings; params: `make`, `model`, `limit` (default 20, max 100), `sort` (`price_asc`/`price_desc`/`newest`) |

- Returns 503 if model not loaded (predict only)
- Returns 404 if no listings in DB (market-stats only)

## Running locally
```bash
# Set environment variables
echo "MONGO_URI=mongodb://localhost:27017" > .env

uvicorn api:app --reload --port 8000
# Then open frontend/index.html in your browser
```

## Pipeline
```bash
python scraper/scraper.py --pages 20
python scraper/scrape_details.py
python ml/preprocess.py
python ml/train.py
python db/migrate.py          # seed MongoDB from CSV (run once)
uvicorn api:app --reload --port 8000
```

## Current model metrics (2,360 rows, with detail pages)
- R²: 0.63
- MAE: LKR 1,480,000
- MAPE: 37.76%

## Do not modify
- ml/preprocess.py feature engineering logic (model was trained on this exact schema)
- Label encoder usage pattern in ml/predict.py (api.py must mirror this exactly)

## PR Reviews

### PR #1 — PriceLens rename + MongoDB API wiring

**Files changed and why**

| File | What changed |
|------|--------------|
| `api.py` | Renamed app title; replaced `@app.on_event("startup")` with lifespan context manager; added `load_dotenv()` at startup; added `GET /listings` and `GET /models/{make}` endpoints; added `comparables` field to `PredictResponse` (fetches up to 5 recent same-make listings from MongoDB after prediction); added `ListingOut` and `ComparableListing` Pydantic models |
| `db/mongo.py` | Renamed default `DB_NAME` from `riyaprice` → `pricelens`; removed broken import artifacts; kept Motor async client and lazy singleton pattern |
| `db/migrate.py` | Renamed default `DB_NAME` from `riyaprice` → `pricelens`; updated docstring |
| `frontend/index.html` | Renamed `RiyaPrice` → `PriceLens` in `<title>`, nav logo, and footer |
| `CLAUDE.md` | Full rewrite: PriceLens name, MongoDB architecture docs, updated endpoint table, updated model metrics (R² 0.63, MAE 1.48M, MAPE 37.76%, 2,360 rows), added env var table and listing document schema |

**Hardcoded values / secrets to move to .env**
- `MONGO_URI` default in `db/migrate.py` falls back to `mongodb://localhost:27017` — acceptable for a CLI migration script, but document that production runs must set `MONGO_URI` explicitly.
- No secrets are committed. `MONGO_URI` is read from env/dotenv in both `db/mongo.py` and `db/migrate.py`.

**Risks and uncertainties**
- `GET /listings` uses `Literal["price_asc","price_desc","newest"]` for the `sort` param — FastAPI validates this at the schema level, but the `sort` field maps to MongoDB field names (`price`, `scraped_at`) which must exist on documents. Listings migrated from CSV will have `scraped_at` set; any future documents without it will sort to the end rather than erroring.
- `POST /predict` comparables query fetches the 5 most recently scraped listings for the same make regardless of year or mileage proximity. This is intentional for now but may surface irrelevant comparables (e.g. a 2005 listing when predicting a 2022 vehicle). A year-range filter (`$gte: year-3, $lte: year+3`) would improve relevance.
- `db/mongo.py` calls `load_dotenv()` at import time and `api.py` lifespan also calls it — this is harmless (dotenv is idempotent) but slightly redundant.
- `GET /models/{make}` returns models distinct from the `model` field; since `migrate.py` currently sets `model: None` for all rows (scraper v2 not yet run), this endpoint will always return an empty list until detail scraping is extended to extract model names. **Resolved by PR #2.**

### PR #2 — feat: extract model field from URL slug for existing listings

**Files changed and why**

| File | What changed |
|------|--------------|
| `db/extract_models.py` | New one-shot sync pymongo script; backfills `model` on all documents where it is `null` by parsing the riyasewana URL slug, with a title-word fallback and a junk filter |

**Hardcoded values / secrets to move to .env**
- `MONGO_URI` and `DB_NAME` both fall back to localhost defaults — same pattern as `db/migrate.py`; acceptable for a CLI one-shot script, must be set explicitly in any non-local environment.

**Risks and uncertainties**
- The URL parser assumes the slug structure `{make}-{model}-sale-{location}-{id}`. If riyasewana ever changes their URL format, the `-sale-` split will miss and the script falls back to the title. That fallback itself requires `make` to be non-null; documents with `make=null` will remain with `model=null` even after running.
- The junk filter only catches exact matches for `{"car", "vehicle", "used", "sale"}` and 4-digit years. Abbreviations, typos, or other placeholder values in the model slot (e.g. "nan", "none", "unknown") would pass through as real model names — none were observed in this run but worth watching in future scrapes.
- The script is idempotent: re-running it is safe because the query filters to `model IS NULL`, so already-filled documents are skipped entirely. However, incorrect extractions (if any were written) cannot be cleaned up by re-running — they'd need a separate correction pass.
- On this run: 2,588/2,588 extracted from URL, 0 title fallbacks, 0 failures. The `GET /models/{make}` endpoint now returns real data.
