"""
predict.py — CLI tool to load the trained model and predict vehicle price.

Usage:
    python ml/predict.py --make Toyota --year 2018 --mileage 65000
    python ml/predict.py   # interactive mode
"""

import argparse
import logging

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CURRENT_YEAR = 2026


def _normalize_condition(raw: str) -> str:
    s = raw.lower()
    if "recondition" in s:
        return "Reconditioned"
    if "unregister" in s:
        return "Unregistered"
    if "brand new" in s or "brandnew" in s:
        return "Brand New"
    if "used" in s:
        return "Used"
    return "Other"


def build_input(bundle: dict, make: str, model_str: str, year: int, mileage: float,
                fuel: str, transmission: str, condition: str, location: str) -> pd.DataFrame:
    row: dict = {
        "year":        year,
        "age":         CURRENT_YEAR - year,
        "log_mileage": np.log1p(mileage),
    }

    if "make_model_means" in bundle:
        key = (make, model_str)
        row["make_model_mean_price"] = bundle["make_model_means"].get(
            key, bundle.get("global_mean_price", 0.0)
        )

    for col, val in [("make", make), ("model", model_str), ("location", location)]:
        if col in bundle["label_encoders"]:
            le = bundle["label_encoders"][col]
            try:
                row[col] = int(le.transform([val])[0])
            except ValueError:
                log.warning(f"Unseen {col} value '{val}' — encoding as -1")
                row[col] = -1

    for feat in bundle["feature_cols"]:
        if feat not in row:
            row[feat] = 0

    for prefix, value in [("fuel_type", fuel), ("transmission", transmission), ("condition", _normalize_condition(condition))]:
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1

    return pd.DataFrame([row])[bundle["feature_cols"]]


def predict(model_path: str, vehicle: dict) -> dict:
    bundle = joblib.load(model_path)

    X = build_input(
        bundle       = bundle,
        make         = vehicle.get("make", "Unknown"),
        model_str    = vehicle.get("model", "Unknown") or "Unknown",
        year         = vehicle["year"],
        mileage      = vehicle["mileage"],
        fuel         = vehicle.get("fuel", "Petrol"),
        transmission = vehicle.get("transmission", "Automatic"),
        condition    = vehicle.get("condition", "Reconditioned"),
        location     = vehicle.get("location", "Unknown"),
    )

    raw_pred = float(bundle["model"].predict(X)[0])
    price = float(np.expm1(raw_pred) if bundle.get("log_transform") else raw_pred)

    return {
        "predicted_price": price,
        "range_low":       price * 0.88,
        "range_high":      price * 1.12,
    }


def interactive_mode(model_path: str):
    """Prompt the user for vehicle details in the terminal."""
    print("\n── Vehicle Price Predictor (riyasewana.com model) ──\n")

    try:
        make         = input("Make (e.g. Toyota, Honda): ").strip() or "Toyota"
        model_name   = input("Model (e.g. Aqua, Vezel) [optional]: ").strip() or "Unknown"
        year         = int(input("Year (e.g. 2018): ").strip())
        mileage      = float(input("Mileage in km (e.g. 65000): ").strip())
        fuel         = input("Fuel type [Petrol/Diesel/Hybrid/Electric] (default: Petrol): ").strip() or "Petrol"
        transmission = input("Transmission [Automatic/Manual] (default: Automatic): ").strip() or "Automatic"
        condition    = input("Condition [Reconditioned/Used/Unregistered/Brand New] (default: Reconditioned): ").strip() or "Reconditioned"
        location     = input("Location (e.g. Colombo) [optional]: ").strip() or "Unknown"
    except KeyboardInterrupt:
        print("\nCancelled.")
        return

    vehicle = {
        "make": make, "model": model_name, "year": year, "mileage": mileage,
        "fuel": fuel, "transmission": transmission,
        "condition": condition, "location": location,
    }

    result = predict(model_path, vehicle)

    print("\n── Prediction ─────────────────────────────────")
    print(f"  Estimated Price : LKR {result['predicted_price']:>12,.0f}")
    print(f"  Likely Range    : LKR {result['range_low']:>12,.0f}  –  LKR {result['range_high']:,.0f}")
    print("────────────────────────────────────────────────\n")


def main():
    parser = argparse.ArgumentParser(description="Predict vehicle price")
    parser.add_argument("--model",        default="models/price_model.joblib")
    parser.add_argument("--make",         type=str)
    parser.add_argument("--model-name",   type=str, default="Unknown")
    parser.add_argument("--year",         type=int)
    parser.add_argument("--mileage",      type=float)
    parser.add_argument("--fuel",         type=str, default="Petrol")
    parser.add_argument("--transmission", type=str, default="Automatic")
    parser.add_argument("--condition",    type=str, default="Reconditioned")
    parser.add_argument("--location",     type=str, default="Unknown")
    args = parser.parse_args()

    if args.make and args.year and args.mileage:
        vehicle = {
            "make": args.make, "model": args.model_name, "year": args.year,
            "mileage": args.mileage, "fuel": args.fuel,
            "transmission": args.transmission, "condition": args.condition,
            "location": args.location,
        }
        result = predict(args.model, vehicle)
        print(f"\nPredicted Price : LKR {result['predicted_price']:,.0f}")
        print(f"Range           : LKR {result['range_low']:,.0f}  –  LKR {result['range_high']:,.0f}\n")
    else:
        interactive_mode(args.model)


if __name__ == "__main__":
    main()