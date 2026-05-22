# Vehicle Price Predictor — Claude Code Context

## Project overview
A full-stack Sri Lankan used vehicle price predictor. Scrapes riyasewana.com, trains an XGBoost model, and serves predictions via FastAPI + a vanilla HTML/CSS/JS dashboard.

## Stack
- **Scraping**: requests + BeautifulSoup
- **ML**: pandas, scikit-learn, XGBoost, SHAP, joblib
- **Backend**: FastAPI + uvicorn
- **Frontend**: Vanilla HTML/CSS/JS (no build step)
- **Python**: 3.13 on Windows

## Project structure
```
RiyaPrice/
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
├── data/
│   ├── raw_listings.csv         # scraped data (gitignored)
│   └── processed.csv            # cleaned, encoded, model-ready (gitignored)
├── models/
│   └── price_model.joblib       # saved bundle (gitignored)
└── outputs/plots/               # actual_vs_predicted.png, feature_importance.png, shap_summary.png
```

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
- Loads model once on startup via `@app.on_event("startup")`
- `POST /predict` → takes PredictRequest, returns PredictResponse
- `GET /market-stats` → returns summary stats from data/processed.csv
- `GET /makes` → returns list of known makes from label_encoders
- `GET /` → health check
- Returns 503 if model not loaded

## Running locally
```bash
uvicorn api:app --reload --port 8000
# Then open frontend/index.html in your browser
```

## Pipeline
```bash
python scraper/scraper.py --pages 20
python scraper/scrape_details.py
python ml/preprocess.py
python ml/train.py
uvicorn api:app --reload --port 8000
```

## Current model metrics (167 rows, no detail pages)
- R²: 0.56
- MAE: LKR 1,630,567
- MAPE: 43.74%
(Will improve significantly once scrape_details.py finishes filling in fuel/transmission/condition)

## Do not modify
- ml/preprocess.py feature engineering logic (model was trained on this exact schema)
- Label encoder usage pattern in ml/predict.py (api.py must mirror this exactly)
