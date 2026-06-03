# PriceLens

> Predict used vehicle prices in Sri Lanka — scrape riyasewana.com, train an XGBoost model, and query predictions through a FastAPI backend with a live market dashboard.

![Python](https://img.shields.io/badge/python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-Motor-47A248?logo=mongodb)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## ✨ Features

- **Price prediction** — XGBoost model trained on 2,500+ real listings; returns a predicted price + ±12% confidence range
- **Comparable listings** — every prediction surfaces up to 5 real same-make listings from MongoDB for context
- **Market dashboard** — live stats (avg, median, min/max price, top makes, fuel distribution) from the database
- **Full pipeline** — scrape → detail-fill → preprocess → train → serve, all scriptable via `make`
- **Filterable listings API** — paginated, filterable by make/model, sortable by price or recency

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your MongoDB connection string
echo "MONGO_URI=mongodb://localhost:27017" > .env

# 3. Start the API (model must be trained first — see Pipeline below)
uvicorn api:app --reload --port 8000
```

Then open `frontend/index.html` in your browser.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Scraping | `requests` + `BeautifulSoup4` |
| ML | `pandas`, `scikit-learn`, `XGBoost`, `SHAP`, `joblib` |
| Backend | `FastAPI` + `uvicorn` |
| Database | `MongoDB` via `Motor` async driver |
| Frontend | Vanilla HTML / CSS / JS (no build step) |
| Python | 3.13 on Windows |

---

## 📦 Installation

**Prerequisites:** Python 3.13, a running MongoDB instance (local or Atlas)

```bash
git clone <repo-url>
cd PriceLens
pip install -r requirements.txt
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | *(required)* | MongoDB connection string |
| `DB_NAME` | `pricelens` | MongoDB database name |

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb://localhost:27017
DB_NAME=pricelens
```

---

## 🏃 Usage

### Full Pipeline

Run these steps in order to go from zero to a working API:

```bash
# 1. Scrape listings from riyasewana.com
make scrape           # ~20 pages by default → data/raw_listings.csv

# 2. Fill in fuel type, transmission, and condition from detail pages
make details

# 3. Clean and engineer features
make preprocess

# 4. Train the XGBoost model
make train            # → models/price_model.joblib + outputs/plots/

# 5. Seed MongoDB from the CSV (run once)
python db/migrate.py

# 6. (Optional) Backfill the model field from URL slugs
python db/extract_models.py

# 7. Serve
make serve            # uvicorn on :8000
```

### Predict from the CLI

```bash
python ml/predict.py --make Toyota --year 2018 --mileage 85000 --fuel Petrol
```

### API

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check; reports `model_loaded` |
| `POST` | `/predict` | Predict price → returns price, ±12% range, 5 comparables |
| `GET` | `/makes` | Distinct active makes from MongoDB |
| `GET` | `/models/{make}` | Distinct active models for a given make |
| `GET` | `/market-stats` | Aggregate price stats, top makes, fuel distribution |
| `GET` | `/listings` | Paginated listings; params: `make`, `model`, `limit`, `sort` |

**Predict request example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "year": 2018,
    "mileage": 85000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "condition": "Registered (Used)",
    "location": "Colombo"
  }'
```

**Predict response:**

```json
{
  "predicted_price": 8500000,
  "range_low": 7480000,
  "range_high": 9520000,
  "currency": "LKR",
  "inputs_used": { "..." : "..." },
  "comparables": [ "..." ]
}
```

**Listings query example:**

```
GET /listings?make=Toyota&sort=price_asc&limit=10
```

`sort` accepts: `price_asc`, `price_desc`, `newest`

---

## 📁 Project Structure

```
PriceLens/
├── api.py                       # FastAPI app (entry point for uvicorn)
├── requirements.txt
├── Makefile
│
├── scraper/
│   ├── scraper.py               # scrapes riyasewana.com search pages
│   └── scrape_details.py        # fills fuel/transmission/condition from detail pages
│
├── ml/
│   ├── preprocess.py            # cleans raw CSV, engineers features
│   ├── train.py                 # trains XGBoost, writes model + SHAP plots
│   └── predict.py               # CLI prediction tool
│
├── frontend/
│   └── index.html               # dashboard UI (open directly in browser)
│
├── db/
│   ├── mongo.py                 # Motor async client, lazy singleton
│   ├── migrate.py               # one-shot CSV → MongoDB bulk upsert
│   └── extract_models.py        # backfills model field from URL slugs
│
├── data/                        # gitignored CSVs
├── models/                      # gitignored model bundle
└── outputs/plots/               # actual_vs_predicted, feature_importance, shap_summary
```

---

## 🏗 Architecture

```
riyasewana.com
      │
      ▼
 scraper.py ──► raw_listings.csv
      │
scrape_details.py (fuel/transmission/condition)
      │
 preprocess.py ──► processed.csv
      │
   train.py ──► price_model.joblib
      │
   migrate.py ──► MongoDB (pricelens.listings)
      │
   api.py (FastAPI)
      │
 frontend/index.html
```

The model bundle (`models/price_model.joblib`) is a `joblib`-serialised dict:

```python
{
  "model":          XGBRegressor,
  "label_encoders": {"make": LabelEncoder, "location": LabelEncoder},
  "feature_cols":   [...],   # ordered; api.py must match exactly
  "metrics":        {"MAE": ..., "RMSE": ..., "R²": ..., "MAPE (%)": ...},
  "log_transform":  True,    # predictions are back-transformed with np.exp()
}
```

**Feature engineering** (stays in sync between `ml/preprocess.py` and `api.py`):
- `age = 2026 - year`
- `log_mileage = np.log1p(mileage)`
- `make`, `location` → LabelEncoded (unseen values → `-1`)
- `fuel_type`, `transmission`, `condition` → one-hot with prefix (e.g. `fuel_type_Hybrid`)

---

## 📊 Current Model Performance

Trained on **2,360 listings** scraped from riyasewana.com:

| Metric | Value |
|---|---|
| R² | 0.63 |
| MAE | LKR 1,480,000 |
| MAPE | 37.76% |
| Confidence band | ±12% |

Prices are in **LKR (Sri Lankan Rupees)**.

---

## 📄 License

MIT
