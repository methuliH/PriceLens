.PHONY: scrape details preprocess train serve clean

scrape:
	python scraper/scraper.py --pages 20 --output data/raw_listings.csv

details:
	python scraper/scrape_details.py --input data/raw_listings.csv

preprocess:
	python ml/preprocess.py --input data/raw_listings.csv --output data/processed.csv

train:
	python ml/train.py --input data/processed.csv --model-out models/price_model.joblib

serve:
	python -m uvicorn api:app --reload --port 8000

clean:
	del /q data\processed.csv models\price_model.joblib outputs\plots\*.png 2>nul || true
