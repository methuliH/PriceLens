# RiyaPrice

End-to-end used car price predictor for the Sri Lankan market. Scrapes 3,500+ live listings from **riyasewana.com**, trains an **XGBoost** regression model with **SHAP** explainability, and serves predictions through a **FastAPI** backend with a polished dashboard UI.

## Quick start

```bash
pip install -r requirements.txt

# Start the API (pre-trained model included)
uvicorn api:app --reload --port 8000

# Open the dashboard — just open frontend/index.html in your browser
```

## Pipeline

Run these steps in order to train a fresh model from scratch:

| Step | Command | Output |
|------|---------|--------|
| 1. Scrape listings | `python scraper/scraper.py --pages 20` | `data/raw_listings.csv` |
| 2. Enrich details | `python scraper/scrape_details.py` | updates `raw_listings.csv` in-place |
| 3. Preprocess | `python ml/preprocess.py` | `data/processed.csv` |
| 4. Train | `python ml/train.py` | `models/price_model.joblib` + SHAP plots |
| 5. Serve | `uvicorn api:app --reload --port 8000` | API on `localhost:8000` |

```bash
# Or use make
make scrape && make details && make preprocess && make train && make serve
```

## Project structure

```
RiyaPrice/
├── api.py                       # FastAPI backend
├── requirements.txt
├── Makefile
│
├── scraper/
│   ├── scraper.py               # search-page scraper (riyasewana.com)
│   └── scrape_details.py        # detail-page enrichment (fuel/trans/condition)
│
├── ml/
│   ├── preprocess.py            # clean raw CSV, engineer features
│   ├── train.py                 # train XGBoost + SHAP plots
│   └── predict.py               # CLI prediction tool
│
├── frontend/
│   └── index.html               # dashboard UI — no build step
│
├── data/                        # gitignored — regenerate with pipeline
├── models/                      # gitignored — regenerate with ml/train.py
└── outputs/plots/               # gitignored — SHAP + eval plots saved here
```

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/makes` | List of known vehicle makes |
| `GET` | `/market-stats` | Avg/median/min/max price + fuel distribution |
| `POST` | `/predict` | Predict price for a vehicle |

**POST /predict** — example request:

```json
{
  "make": "Toyota",
  "year": 2018,
  "mileage": 65000,
  "fuel_type": "Hybrid",
  "transmission": "Automatic",
  "condition": "Unregistered",
  "location": "Colombo"
}
```

**Response:**

```json
{
  "predicted_price": 7850000,
  "range_low": 6908000,
  "range_high": 8792000,
  "currency": "LKR",
  "inputs_used": { "..." : "..." }
}
```

## Model

| | |
|---|---|
| Algorithm | XGBoost, log-scale target (`log1p`) |
| R² | 0.56 |
| MAE | LKR 1,630,567 |
| MAPE | 43.74% |
| Training rows | 167 (detail enrichment pending) |
| Confidence band | ± 12% |

Accuracy will improve significantly once `scrape_details.py` finishes enriching fuel/transmission/condition for all listings.

## Tech stack

- **Scraping** — `requests` + `BeautifulSoup4`
- **ML** — `pandas`, `scikit-learn`, `XGBoost`, `SHAP`, `joblib`
- **Backend** — `FastAPI` + `uvicorn`
- **Frontend** — Vanilla HTML / CSS / JS (no build step)
- **Python** — 3.13
