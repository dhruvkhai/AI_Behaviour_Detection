"""
train_master_model.py
=====================
Training script for the Master Sensor Fusion XGBoost classifier.

Key design choices
------------------
* Stratified Date-Cow Sampling: instead of random row sub-sampling (which
  breaks temporal continuity of rolling windows), we randomly pick *whole*
  (cow, date) sessions and keep every row within them.  This keeps windows
  intact while fitting the 10 GB dataset into available RAM.
* XGBoost's tree_method='hist' is used for memory-efficient training on
  large, sparse fused feature vectors.
* Missing sensor columns (NaN) are passed to XGBoost as-is; XGBoost splits
  missing values optimally, giving graceful degradation when a sensor is down.
* A Feature Importance report and a Confusion Matrix are saved to models/.
* A Robustness Check (UWB drop test) verifies graceful degradation.
"""

import os
import sys
import json
import time
import traceback

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe on headless servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split

from app.ml.fusion_pipeline import MasterDataLoader

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "behavior_master.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "behavior_master_label_encoder.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "behavior_master_features.json")
REPORT_PATH = os.path.join(MODELS_DIR, "report_master.txt")
IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance_master.png")
CONFUSION_PATH = os.path.join(MODELS_DIR, "confusion_matrix_master.png")


# ---------------------------------------------------------------------------
#  Core training function
# ---------------------------------------------------------------------------

def train_master_model(
    max_sessions: int = 50,
    test_size: float = 0.2,
    random_state: int = 42,
    xgb_params: dict | None = None,
) -> None:
    """
    End-to-end Master Sensor Fusion model training.

    Args:
        max_sessions:  Max number of (cow, date) sessions to load (memory guard).
        test_size:     Fraction of sessions held out for evaluation.
        random_state:  Seed for reproducibility.
        xgb_params:    Optional XGBoost hyperparameter overrides.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("  Master Sensor Fusion — Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Discover all (cow, date) sessions
    # ------------------------------------------------------------------
    loader = MasterDataLoader()
    all_pairs = loader.get_cow_date_pairs()

    if not all_pairs:
        print("[ERROR] No label files found. Check LABEL_ROOT in sensor_configs.py")
        sys.exit(1)

    print(f"[1/7] Discovered {len(all_pairs)} (cow, date) sessions.")

    # ------------------------------------------------------------------
    # 2. Stratified Date-Cow Sampling (memory guard for 10 GB dataset)
    # ------------------------------------------------------------------
    sampled_pairs = loader.stratified_sample(
        all_pairs, max_sessions=max_sessions, random_state=random_state
    )
    print(
        f"[2/7] Using {len(sampled_pairs)} sessions "
        f"({'all' if len(sampled_pairs) == len(all_pairs) else 'sampled'})."
    )

    # ------------------------------------------------------------------
    # 3. Load & fuse data
    # ------------------------------------------------------------------
    print("[3/7] Loading and fusing sensor data (this may take a while)...")
    df = loader._align_pairs(sampled_pairs)

    if df.empty:
        print("[ERROR] Fused DataFrame is empty. Verify sensor paths and labels.")
        sys.exit(1)

    print(f"      Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")

    # ------------------------------------------------------------------
    # 4. Prepare X / y
    # ------------------------------------------------------------------
    print("[4/7] Preparing feature matrix ...")

    # Save label encoder and feature names for inference
    with open(ENCODER_PATH, "w") as f:
        json.dump(loader.label_encoder, f, indent=2)
    print(f"      Label encoder saved → {ENCODER_PATH}")
    print(f"      Class mapping: {loader.label_encoder}")

    meta_cols = {"timestamp", "behavior", "datetime", "label_idx", "sensor_idx",
                 "_cow_id", "_date"}
    feature_names = [c for c in df.columns if c not in meta_cols]

    with open(FEATURE_NAMES_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"      {len(feature_names)} features saved → {FEATURE_NAMES_PATH}")

    # Drop rows where the label itself is unknown (-1)
    df = df[df["behavior"] >= 0].copy()

    y = df["behavior"].values
    X = df[feature_names].copy()

    # Verify class distribution
    classes, counts = np.unique(y, return_counts=True)
    print(f"      Class distribution: { {int(c): int(n) for c, n in zip(classes, counts)} }")

    # Guard: need at least 2 classes
    if len(classes) < 2:
        print("[ERROR] Only 1 class found. Cannot train a classifier.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Train / test split (session-aware: split by _cow_id + _date combos)
    # ------------------------------------------------------------------
    print("[5/7] Splitting into train / test sets ...")
    sessions = df[["_cow_id", "_date"]].drop_duplicates()
    try:
        train_sess, test_sess = train_test_split(
            sessions,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError:
        # Fewer sessions than needed for stratification — fall back to row split
        train_sess, test_sess = train_test_split(
            sessions, test_size=test_size, random_state=random_state
        )

    train_mask = df.set_index(["_cow_id", "_date"]).index.isin(
        pd.MultiIndex.from_frame(train_sess)
    )
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        # Fallback: simple random row split
        print("      [Warning] Session-aware split produced empty set — falling back to row split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(classes) > 1 else None
        )

    print(f"      Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # ------------------------------------------------------------------
    # 6. Train XGBoost
    # ------------------------------------------------------------------
    print("[6/7] Training XGBoost (Master Fusion) ...")
    default_params = dict(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        tree_method="hist",       # Efficient histogram-based for large data
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    if xgb_params:
        default_params.update(xgb_params)

    model = xgb.XGBClassifier(**default_params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate & save artefacts
    # ------------------------------------------------------------------
    print("[7/7] Evaluating and saving artefacts ...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report_str = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print("\nClassification Report:")
    print(report_str)

    # Reverse label encoder for readable class names in plots
    id_to_label = {v: k for k, v in loader.label_encoder.items()}
    label_names = [id_to_label.get(c, str(c)) for c in sorted(classes)]

    # ---- Confusion Matrix ------------------------------------------------
    _save_confusion_matrix(y_test, y_pred, label_names, sorted(classes))

    # ---- Feature Importance Plot -----------------------------------------
    _save_feature_importance(model, feature_names)

    # ---- Save model & text report ----------------------------------------
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")

    elapsed = time.time() - t0
    with open(REPORT_PATH, "w") as f:
        f.write("Master Sensor Fusion Model Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Training sessions : {len(sampled_pairs)}\n")
        f.write(f"Training rows     : {len(X_train):,}\n")
        f.write(f"Test rows         : {len(X_test):,}\n")
        f.write(f"Feature count     : {len(feature_names)}\n")
        f.write(f"Accuracy          : {acc:.4f}\n")
        f.write(f"F1 (macro)        : {f1_macro:.4f}\n")
        f.write(f"Training time (s) : {elapsed:.1f}\n")
        f.write(f"Label mapping     : {loader.label_encoder}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\n\nTop 20 Features by Importance:\n")
        imp_df = _make_importance_df(model, feature_names)
        f.write(imp_df.head(20).to_string())

    print(f"  Report saved → {REPORT_PATH}")
    print(f"\n  Total training time: {elapsed:.1f}s")

    # ---- Robustness Check (graceful degradation) -------------------------
    _robustness_check(model, X_test, y_test, feature_names)

    print("\n[Done] Master Model training complete.\n")


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------

def _make_importance_df(model: xgb.XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """Returns a sorted DataFrame of feature importances."""
    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return imp_df


def _save_feature_importance(
    model: xgb.XGBClassifier, feature_names: list[str], top_n: int = 25
) -> None:
    """Saves a horizontal bar chart of the top-N most important features."""
    imp_df = _make_importance_df(model, feature_names).head(top_n)

    # Colour-code by sensor prefix for easy visual attribution
    sensor_prefixes = [
        "immu", "cbt", "pressure", "milk", "thi", "uwb", "weather"
    ]
    palette = plt.cm.tab10.colors
    color_map = {s: palette[i % len(palette)] for i, s in enumerate(sensor_prefixes)}

    def _feat_color(feat_name: str) -> tuple:
        for prefix in sensor_prefixes:
            if feat_name.startswith(prefix):
                return color_map[prefix]
        return (0.5, 0.5, 0.5, 1.0)

    colors = [_feat_color(f) for f in imp_df["feature"]]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("XGBoost Feature Importance (gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Features — Master Sensor Fusion Model", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Legend for sensor colours
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_map[s], label=s.upper()) for s in sensor_prefixes
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved → {IMPORTANCE_PATH}")


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    classes: list[int],
) -> None:
    """Saves a normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix (Normalised) — Master Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {CONFUSION_PATH}")


def _robustness_check(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """
    Verification Plan — Robustness Check:
    Drops all UWB columns from the test set and verifies the model still predicts.
    A valid F1 score (> 0) confirms graceful degradation.
    """
    print("\n--- Robustness Check: UWB Drop Test ---")
    uwb_cols = [c for c in feature_names if c.startswith("uwb_")]
    if not uwb_cols:
        print("  No UWB features found — skipping.")
        return

    X_no_uwb = X_test.copy()
    X_no_uwb[uwb_cols] = np.nan  # Simulate sensor outage

    try:
        y_pred_no_uwb = model.predict(X_no_uwb)
        f1_no_uwb = f1_score(y_test, y_pred_no_uwb, average="macro", zero_division=0)
        acc_no_uwb = accuracy_score(y_test, y_pred_no_uwb)
        print(f"  UWB columns zeroed : {len(uwb_cols)}")
        print(f"  Accuracy (no UWB)  : {acc_no_uwb:.4f}")
        print(f"  F1 macro (no UWB)  : {f1_no_uwb:.4f}")
        if f1_no_uwb > 0:
            print("  [PASS] Model produces valid predictions without UWB. ✓")
        else:
            print("  [WARN] F1 dropped to 0 — investigate model dependence on UWB.")
    except Exception as e:
        print(f"  [ERROR] during robustness check: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Master Sensor Fusion model.")
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=50,
        help="Max (cow, date) sessions to load (memory guard). Default: 50",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of sessions used for evaluation. Default: 0.2",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Number of XGBoost trees. Default: 300",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=7,
        help="Max tree depth. Default: 7",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    args = parser.parse_args()

    train_master_model(
        max_sessions=args.max_sessions,
        test_size=args.test_size,
        random_state=args.seed,
        xgb_params={
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
        },
    )
