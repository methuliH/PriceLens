#Load the trained model and predict the price

import argparse
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CURRENT_YEAR = 2026


def build_input(make, year, mileage, fuel, transmission, condition, location, feature_cols, label_encoders):
    """
    Build a single-row DataFrame that matches the model's expected feature schema.
    """
    age        = CURRENT_YEAR - year
    log_mileage = np.log1p(mileage)

    row = {
        "year":        year,
        "age":         age,
        "log_mileage": log_mileage,
    }

    # Label-encode make and location if encoders exist
    for col, val in [("make", make), ("location", location)]:
        if col in label_encoders:
            le = label_encoders[col]
            try:
                row[col] = le.transform([val])[0]
            except ValueError:
                # Unseen label — use -1 (model will treat as unknown)
                log.warning(f"'{val}' is an unseen {col}. Using fallback encoding.")
                row[col] = -1

    # One-hot encoded columns (set all to 0, then flip matching ones)
    for feat in feature_cols:
        if feat not in row:
            row[feat] = 0

    # Flip the right OHE columns
    ohe_map = {
        "fuel_type":    fuel,
        "transmission": transmission,
        "condition":    condition,
    }
    for prefix, value in ohe_map.items():
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1

    df = pd.DataFrame([row])[feature_cols]
    return df


def predict(model_path: str, vehicle: dict) -> dict:
    bundle = joblib.load(model_path)
    model          = bundle["model"]
    label_encoders = bundle["label_encoders"]
    feature_cols   = bundle["feature_cols"]

    X = build_input(
        make         = vehicle.get("make", "Unknown"),
        year         = vehicle["year"],
        mileage      = vehicle["mileage"],
        fuel         = vehicle.get("fuel", "Petrol"),
        transmission = vehicle.get("transmission", "Automatic"),
        condition    = vehicle.get("condition", "Reconditioned"),
        location     = vehicle.get("location", "Unknown"),
        feature_cols = feature_cols,
        label_encoders = label_encoders,
    )

    price = float(np.exp(model.predict(X)[0]))

    # Rough confidence band: ± 10% (replace with proper quantile regression later)
    low  = price * 0.90
    high = price * 1.10

    return {
        "predicted_price": price,
        "range_low":       low,
        "range_high":      high,
    }


def interactive_mode(model_path: str):
    """Prompt the user for vehicle details in the terminal."""
    print("\n── Vehicle Price Predictor (riyasewana.com model) ──\n")

    try:
        make         = input("Make (e.g. Toyota, Honda): ").strip() or "Toyota"
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
        "make": make, "year": year, "mileage": mileage,
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
    parser.add_argument("--year",         type=int)
    parser.add_argument("--mileage",      type=float)
    parser.add_argument("--fuel",         type=str, default="Petrol")
    parser.add_argument("--transmission", type=str, default="Automatic")
    parser.add_argument("--condition",    type=str, default="Reconditioned")
    parser.add_argument("--location",     type=str, default="Unknown")
    args = parser.parse_args()

    # If all required args provided, run non-interactively
    if args.make and args.year and args.mileage:
        vehicle = {
            "make": args.make, "year": args.year, "mileage": args.mileage,
            "fuel": args.fuel, "transmission": args.transmission,
            "condition": args.condition, "location": args.location,
        }
        result = predict(args.model, vehicle)
        print(f"\nPredicted Price : LKR {result['predicted_price']:,.0f}")
        print(f"Range           : LKR {result['range_low']:,.0f}  –  LKR {result['range_high']:,.0f}\n")
    else:
        interactive_mode(args.model)


if __name__ == "__main__":
    main()