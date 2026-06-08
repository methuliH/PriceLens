<h1 align="center">PriceLens</h1>
<p align="center">Sri Lankan used vehicle price predictor — scrape, train, and predict in one pipeline.</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python 3.13">
  <img src="https://img.shields.io/github/last-commit/methuliH/PriceLens" alt="Last commit">
  <img src="https://img.shields.io/github/stars/methuliH/PriceLens?style=social" alt="Stars">
</p>

PriceLens scrapes live listings from **riyasewana.com**, trains an **XGBoost** regression model with **SHAP** explainability, and serves price predictions through a **FastAPI** backend backed by **MongoDB**. A vanilla HTML/CSS/JS dashboard lets users get instant estimates with a ±12% confidence range and comparable listings pulled live from the database.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## Features

- Scrapes 2,360+ riyasewana.com listings including detail-page enrichment for fuel type, transmission, and condition
- XGBoost model trained on log-scale prices with 5-fold cross-validation and early stopping
- Per-(make, model) mean price encoding as a target-leak-free feature
- Predictions served with a **±12% confidence band** and up to 5 live comparable listings
- SHAP summary, feature importance, and actual-vs-predicted plots saved on every training run
- Full MongoDB persistence — all API endpoints read live from the database, no CSV at runtime
- Paginated `/listings` endpoint with `make`, `model`, `fuel_type` filters and `price_asc / price_desc / newest` sort

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Scraping | requests + BeautifulSoup4 |
| ML | XGBoost, scikit-learn, SHAP, joblib |
| Backend | FastAPI + uvicorn |
| Database | MongoDB via Motor (async) |
| Frontend | Vanilla HTML / CSS / JS (no build step) |
| Python | 3.13 |

## Getting Started

### Prerequisites

- Python 3.13
- A running MongoDB instance (local or Atlas)

### Installation

```bash
git clone https://github.com/methuliH/PriceLens.git
cd PriceLens
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```bash
echo "MONGO_URI=mongodb://localhost:27017" > .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | *(required)* | MongoDB connection string |
| `DB_NAME` | `pricelens` | MongoDB database name |

## Usage

### Quick start (pre-trained model)

```bash
uvicorn api:app --reload --port 8000
```

Then open `frontend/index.html` in your browser.

### Full pipeline (train from scratch)

Run these steps in order:

| Step | Command | Output |
|------|---------|--------|
| 1. Scrape listings | `python scraper/scraper.py --pages 20` | `data/raw_listings.csv` |
| 2. Enrich details | `python scraper/scrape_details.py` | updates `raw_listings.csv` in-place |
| 3. Preprocess | `python ml/preprocess.py` | `data/processed.csv` |
| 4. Train | `python ml/train.py` | `models/price_model.joblib` + plots |
| 5. Seed MongoDB | `python db/migrate.py` | upserts all rows into `listings` collection |
| 6. Serve | `uvicorn api:app --reload --port 8000` | API on `localhost:8000` |

Or use `make`:

```bash
make scrape && make details && make preprocess && make train && make serve
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check; reports `model_loaded` status |
| `GET` | `/makes` | Distinct active vehicle makes from MongoDB |
| `GET` | `/models/{make}` | Distinct active models for a given make |
| `GET` | `/market-stats` | Aggregate price stats, top 5 makes, fuel distribution |
| `GET` | `/listings` | Paginated active listings with filters and sort |
| `POST` | `/predict` | Predict price → returns estimate, ±12% range, 5 comparables |

### GET /listings — query params

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `make` | string | — | Filter by make |
| `model` | string | — | Filter by model |
| `fuel_type` | string | — | Filter by fuel type |
| `limit` | int | 20 | Max results (1–100) |
| `sort` | string | `newest` | `price_asc`, `price_desc`, or `newest` |

### POST /predict

<details>
<summary>Request / Response example</summary>

**Request:**
```json
{
  "make": "Toyota",
  "model": "Prius",
  "year": 2019,
  "mileage": 60000,
  "fuel_type": "Hybrid",
  "transmission": "Automatic",
  "condition": "Registered (Used)",
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
  "inputs_used": { "...": "..." },
  "comparables": [
    {
      "title": "Toyota Prius 2019",
      "make": "Toyota",
      "year": 2019,
      "mileage": 58000,
      "price": 7500000,
      "fuel_type": "Hybrid",
      "transmission": "Automatic",
      "condition": "Registered (Used)",
      "url": "https://riyasewana.com/..."
    }
  ]
}
```
</details>

Returns `503` if the model is not loaded. Returns `404` from `/market-stats` if the database has no listings.

## Model Performance

Trained on **2,360 rows** with detail-page enrichment (fuel type, transmission, condition, engine cc).

| Metric | Value |
|--------|-------|
| Algorithm | XGBoost — log-scale target (`log1p`) |
| R² | 0.63 |
| MAE | LKR 1,480,000 |
| MAPE | 37.76% |
| Confidence band | ± 12% |

Training outputs three plots to `outputs/plots/`: `actual_vs_predicted.png`, `feature_importance.png`, `shap_summary.png`.

## Project Structure

```
PriceLens/
├── api.py                    # FastAPI entry point
├── requirements.txt
├── Makefile
├── scraper/
│   ├── scraper.py            # search-page scraper (riyasewana.com)
│   └── scrape_details.py     # detail-page enrichment
├── ml/
│   ├── preprocess.py         # clean CSV, engineer features
│   ├── train.py              # train XGBoost + SHAP plots
│   └── predict.py            # CLI prediction tool
├── db/
│   ├── mongo.py              # Motor async client, lazy singleton
│   └── migrate.py            # one-shot CSV → MongoDB bulk upsert
├── frontend/
│   └── index.html            # dashboard UI — no build step
├── data/                     # gitignored — regenerate with pipeline
├── models/                   # gitignored — regenerate with ml/train.py
└── outputs/plots/            # SHAP + eval plots saved here
```

## Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.
