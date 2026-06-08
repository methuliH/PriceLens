"""
train.py
--------
Trains an XGBoost model to predict vehicle prices.
Reads training data from MongoDB (or falls back to processed.csv).
Outputs: model bundle, feature importance chart, SHAP summary.

Usage:
    python ml/train.py
    python ml/train.py --input data/processed.csv   # force CSV path
    python ml/train.py --model-out models/price_model.joblib
"""

import os
import sys
import logging
import argparse

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Ensure project root is importable when running as a script from any CWD
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.preprocess import load_from_mongo, load_from_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Feature config ────────────────────────────────────────────────────────────

NUMERIC_FEATURES     = ["age", "log_mileage", "year", "engine_cc_num", "make_model_mean_price"]
CATEGORICAL_FEATURES = ["make", "model", "location"]
OHE_PREFIXES         = ["fuel_type_", "transmission_", "condition_"]
TARGET               = "price"


# ── Feature matrix builder ────────────────────────────────────────────────────

def load_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict, list]:
    ohe_cols = [c for c in df.columns if any(c.startswith(p) for p in OHE_PREFIXES)]

    df = df.copy()
    label_encoders: dict[str, LabelEncoder] = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown").astype(str)
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    feature_cols = (
        [c for c in NUMERIC_FEATURES     if c in df.columns]
        + [c for c in CATEGORICAL_FEATURES if c in df.columns]
        + ohe_cols
    )

    X = df[feature_cols].fillna(0)
    y = df[TARGET]
    return X, y, label_encoders, feature_cols


# ── Model training ─────────────────────────────────────────────────────────────

def _fit_xgb(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=50)
    return model


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_raw(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape}
    print("\n-- Model Evaluation (LKR scale) --")
    print(f"  MAE  : LKR {mae:>15,.0f}")
    print(f"  RMSE : LKR {rmse:>15,.0f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("-----------------------------------\n")
    return metrics


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_predictions(y_test, preds, out_dir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, preds, alpha=0.4, s=20, color="#2563eb")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Price (LKR)")
    ax.set_ylabel("Predicted Price (LKR)")
    ax.set_title("Actual vs Predicted Vehicle Price")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "actual_vs_predicted.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved plot -> {path}")


def plot_feature_importance(model, feature_cols: list, out_dir: str):
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feat_df["feature"], feat_df["importance"], color="#2563eb")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Saved plot -> {path}")


def plot_shap(model, X_test, out_dir: str):
    log.info("Computing SHAP values (may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved SHAP plot -> {path}")


# ── Pipeline (callable by scheduler) ──────────────────────────────────────────

def train(
    input_csv: str  = "data/processed.csv",
    model_out: str  = "models/price_model.joblib",
    plot_dir:  str  = "outputs/plots",
    test_size: float = 0.2,
) -> dict:
    """Run the full training pipeline and return evaluation metrics."""
    load_dotenv()

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # ── Load data: MongoDB preferred, CSV fallback ────────────────────────────
    if os.getenv("MONGO_URI"):
        try:
            df = load_from_mongo()
            log.info(f"Loaded from MongoDB: {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as exc:
            log.warning(f"MongoDB load failed ({exc}) -- falling back to CSV: {input_csv}")
            df = load_from_csv(input_csv)
    else:
        log.info(f"MONGO_URI not set -- loading from CSV: {input_csv}")
        df = load_from_csv(input_csv)

    if "model" not in df.columns:
        df["model"] = "Unknown"

    log.info(f"Dataset: {df.shape[0]} rows, {df.shape[1]} cols")
    df = df.reset_index(drop=True)

    # ── Pre-split index to compute group means without leakage ────────────────
    idx_all = list(range(len(df)))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=42
    )

    train_slice = df.iloc[idx_train][["make", "model", "price"]].copy()
    train_slice["make"]  = train_slice["make"].fillna("Unknown").astype(str)
    train_slice["model"] = train_slice["model"].fillna("Unknown").astype(str)

    global_mean: float = float(train_slice["price"].mean())
    make_model_means: dict = (
        train_slice.groupby(["make", "model"])["price"]
        .mean()
        .to_dict()
    )
    log.info(
        f"Computed mean prices for {len(make_model_means)} (make, model) pairs "
        f"(global mean: LKR {global_mean:,.0f})"
    )

    _make  = df["make"].fillna("Unknown").astype(str)
    _model = df["model"].fillna("Unknown").astype(str)
    df["make_model_mean_price"] = [
        make_model_means.get((m, mod), global_mean)
        for m, mod in zip(_make, _model)
    ]

    # ── Build feature matrix ──────────────────────────────────────────────────
    X, y, label_encoders, feature_cols = load_features(df)
    log.info(f"Features ({len(feature_cols)}): {feature_cols}")

    y_log       = np.log1p(y)
    X_train     = X.iloc[idx_train]
    X_test      = X.iloc[idx_test]
    y_log_train = y_log.iloc[idx_train]
    y_log_test  = y_log.iloc[idx_test]
    y_test      = np.expm1(y_log_test)
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    log.info("Running 5-fold cross-validation...")
    cv_model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1
    )
    cv_scores = cross_val_score(
        cv_model, X, y_log, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    log.info(f"CV MAE (log scale): {-cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ── Train final model ─────────────────────────────────────────────────────
    log.info("Training final model...")
    model = _fit_xgb(X_train, y_log_train)

    log_preds = model.predict(X_test)
    preds     = np.expm1(log_preds)
    metrics   = evaluate_raw(y_test, preds)

    plot_predictions(y_test, preds, plot_dir)
    plot_feature_importance(model, feature_cols, plot_dir)
    plot_shap(model, X_test, plot_dir)

    # ── Save bundle ───────────────────────────────────────────────────────────
    bundle = {
        "model":             model,
        "label_encoders":    label_encoders,
        "feature_cols":      feature_cols,
        "metrics":           metrics,
        "log_transform":     True,
        "make_model_means":  make_model_means,
        "global_mean_price": global_mean,
    }
    joblib.dump(bundle, model_out)
    log.info(f"Model saved -> {model_out}")

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/processed.csv",
                        help="CSV fallback path (used only when MONGO_URI is unset)")
    parser.add_argument("--model-out", default="models/price_model.joblib")
    parser.add_argument("--plot-dir",  default="outputs/plots")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    train(
        input_csv=args.input,
        model_out=args.model_out,
        plot_dir=args.plot_dir,
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
