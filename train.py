"""
train.py
--------
Trains an XGBoost model to predict vehicle prices.
Outputs: model file, feature importance chart, SHAP summary.

Usage:
    python train.py --input data/processed.csv --model-out models/price_model.joblib
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Feature config ───────────────────────────────────────────────────────────

NUMERIC_FEATURES = ["age", "log_mileage", "year", "engine_cc_num"]
CATEGORICAL_FEATURES = ["make", "model", "location"]

# One-hot encoded columns (prefix-based)
OHE_PREFIXES = ["fuel_type_", "transmission_", "condition_"]

TARGET = "price"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select and prepare feature matrix X and target y."""

    # Collect OHE columns
    ohe_cols = [c for c in df.columns if any(c.startswith(p) for p in OHE_PREFIXES)]

    # Label-encode categorical features
    df = df.copy()  # avoid SettingWithCopyWarning
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown").astype(object)
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Build final feature list
    feature_cols = (
        [c for c in NUMERIC_FEATURES if c in df.columns]
        + [c for c in CATEGORICAL_FEATURES if c in df.columns]
        + ohe_cols
    )

    X = df[feature_cols].fillna(0)
    y = df[TARGET]

    return X, y, label_encoders, feature_cols


# ── Model training ────────────────────────────────────────────────────────────

def train(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    eval_set = [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}
    print("\n── Model Evaluation ──────────────────────────")
    print(f"  MAE  : LKR {mae:>15,.0f}")
    print(f"  RMSE : LKR {rmse:>15,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("─────────────────────────────────────────────\n")
    return metrics


def evaluate_raw(y_true, y_pred) -> dict:
    """Evaluate in original LKR scale (after back-transforming from log)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}
    print("\n── Model Evaluation (LKR scale) ──────────────")
    print(f"  MAE  : LKR {mae:>15,.0f}")
    print(f"  RMSE : LKR {rmse:>15,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("─────────────────────────────────────────────\n")
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predictions(y_test, preds, out_dir: str):
    """Actual vs Predicted scatter plot."""
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
    log.info(f"Saved plot → {path}")


def plot_feature_importance(model, feature_cols: list, out_dir: str):
    """XGBoost built-in feature importance."""
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
    log.info(f"Saved plot → {path}")


def plot_shap(model, X_test, out_dir: str):
    """SHAP summary plot — shows which features drive predictions."""
    log.info("Computing SHAP values (may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved SHAP plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/processed.csv")
    parser.add_argument("--model-out", default="models/price_model.joblib")
    parser.add_argument("--plot-dir",  default="outputs/plots")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    log.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    X, y, label_encoders, feature_cols = load_features(df)
    log.info(f"Features: {len(feature_cols)}")

    # Log-transform target — compresses the wide price range, improves MAPE significantly
    y_log = np.log1p(y)

    # Train/test split
    X_train, X_test, y_log_train, y_log_test = train_test_split(
        X, y_log, test_size=args.test_size, random_state=42
    )
    y_test = np.expm1(y_log_test)  # original scale for evaluation
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Cross-validation on log scale
    log.info("Running 5-fold cross-validation...")
    cv_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(cv_model, X, y_log, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    log.info(f"CV MAE (log scale): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final model — trained on log prices
    log.info("Training final model...")
    model = train(X_train, y_log_train)

    # Evaluate — back-transform to LKR
    log_preds = model.predict(X_test)
    preds     = np.expm1(log_preds)
    metrics   = evaluate_raw(y_test, preds)

    # Plots
    plot_predictions(y_test, preds, args.plot_dir)
    plot_feature_importance(model, feature_cols, args.plot_dir)
    plot_shap(model, X_test, args.plot_dir)

    # Save model + metadata
    bundle = {
        "model":          model,
        "label_encoders": label_encoders,
        "feature_cols":   feature_cols,
        "metrics":        metrics,
        "log_transform":  True,   # predictions must be back-transformed with np.expm1()
    }
    joblib.dump(bundle, args.model_out)
    log.info(f"Model saved → {args.model_out}")


if __name__ == "__main__":
    main()"""
train.py
--------
Trains an XGBoost model to predict vehicle prices.
Outputs: model file, feature importance chart, SHAP summary.

Usage:
    python train.py --input data/processed.csv --model-out models/price_model.joblib
"""

import pandas as pd
import numpy as np
import argparse
import os
import logging
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Feature config ───────────────────────────────────────────────────────────

NUMERIC_FEATURES = ["age", "log_mileage", "year", "engine_cc_num"]
CATEGORICAL_FEATURES = ["make", "model", "location"]

# One-hot encoded columns (prefix-based)
OHE_PREFIXES = ["fuel_type_", "transmission_", "condition_"]

TARGET = "price"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select and prepare feature matrix X and target y."""

    # Collect OHE columns
    ohe_cols = [c for c in df.columns if any(c.startswith(p) for p in OHE_PREFIXES)]

    # Label-encode categorical features
    df = df.copy()  # avoid SettingWithCopyWarning
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown").astype(object)
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Build final feature list
    feature_cols = (
        [c for c in NUMERIC_FEATURES if c in df.columns]
        + [c for c in CATEGORICAL_FEATURES if c in df.columns]
        + ohe_cols
    )

    X = df[feature_cols].fillna(0)
    y = df[TARGET]

    return X, y, label_encoders, feature_cols


# ── Model training ────────────────────────────────────────────────────────────

def train(X_train, y_train) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,       # L1 regularization
        reg_lambda=1.0,      # L2 regularization
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="mae",
    )

    eval_set = [(X_train, y_train)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}
    print("\n── Model Evaluation ──────────────────────────")
    print(f"  MAE  : LKR {mae:>15,.0f}")
    print(f"  RMSE : LKR {rmse:>15,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("─────────────────────────────────────────────\n")
    return metrics


def evaluate_raw(y_true, y_pred) -> dict:
    """Evaluate in original LKR scale (after back-transforming from log)."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE (%)": mape}
    print("\n── Model Evaluation (LKR scale) ──────────────")
    print(f"  MAE  : LKR {mae:>15,.0f}")
    print(f"  RMSE : LKR {rmse:>15,.0f}")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("─────────────────────────────────────────────\n")
    return metrics


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_predictions(y_test, preds, out_dir: str):
    """Actual vs Predicted scatter plot."""
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
    log.info(f"Saved plot → {path}")


def plot_feature_importance(model, feature_cols: list, out_dir: str):
    """XGBoost built-in feature importance."""
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
    log.info(f"Saved plot → {path}")


def plot_shap(model, X_test, out_dir: str):
    """SHAP summary plot — shows which features drive predictions."""
    log.info("Computing SHAP values (may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test, show=False, max_display=15)
    plt.tight_layout()
    path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved SHAP plot → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     default="data/processed.csv")
    parser.add_argument("--model-out", default="models/price_model.joblib")
    parser.add_argument("--plot-dir",  default="outputs/plots")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.input)
    log.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

    X, y, label_encoders, feature_cols = load_features(df)
    log.info(f"Features: {len(feature_cols)}")

    # Log-transform target — compresses the wide price range, improves MAPE significantly
    y_log = np.log1p(y)

    # Train/test split
    X_train, X_test, y_log_train, y_log_test = train_test_split(
        X, y_log, test_size=args.test_size, random_state=42
    )
    y_test = np.expm1(y_log_test)  # original scale for evaluation
    log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Cross-validation on log scale
    log.info("Running 5-fold cross-validation...")
    cv_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(cv_model, X, y_log, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    log.info(f"CV MAE (log scale): {-cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final model — trained on log prices
    log.info("Training final model...")
    model = train(X_train, y_log_train)

    # Evaluate — back-transform to LKR
    log_preds = model.predict(X_test)
    preds     = np.expm1(log_preds)
    metrics   = evaluate_raw(y_test, preds)

    # Plots
    plot_predictions(y_test, preds, args.plot_dir)
    plot_feature_importance(model, feature_cols, args.plot_dir)
    plot_shap(model, X_test, args.plot_dir)

    # Save model + metadata
    bundle = {
        "model":          model,
        "label_encoders": label_encoders,
        "feature_cols":   feature_cols,
        "metrics":        metrics,
        "log_transform":  True,   # predictions must be back-transformed with np.expm1()
    }
    joblib.dump(bundle, args.model_out)
    log.info(f"Model saved → {args.model_out}")


if __name__ == "__main__":
    main()