"""
train_master_model.py
=====================
Train the main XGBoost sensor-fusion model (the one the API uses).

Run (all data, GPU if available):
  python train_master_model.py --max-sessions 0 --top-features 50
"""

import os
import sys
import json
import time
import traceback
import argparse

import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from app.ml.fusion_pipeline import MasterDataLoader
from app.ml.training_utils import (
    MERGED_CLASS_ID,
    compute_sample_weights,
    merge_weak_classes,
    recode_labels_consecutive,
    resolve_sessions,
    save_metrics_report,
    session_train_test_split,
    top_features_by_importance,
    xgboost_device,
    xgboost_predict,
)

# Fix import - I used _make_importance_df_helper but defined _make_importance_df in original. I'll use local _make_importance_df in master file.

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "behavior_master.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "behavior_master_label_encoder.json")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "behavior_master_features.json")
REPORT_PATH = os.path.join(MODELS_DIR, "report_master.txt")
IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance_master.png")
CONFUSION_PATH = os.path.join(MODELS_DIR, "confusion_matrix_master.png")
SELECTED_FEATURES_PATH = os.path.join(MODELS_DIR, "behavior_master_selected_features.json")


def _make_importance_df(model, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)


def train_master_model(
    max_sessions: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
    top_features: int = 50,
    merge_weak: bool = True,
    xgb_gpu: bool = False,
    xgb_params: dict | None = None,
) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("  Master Sensor Fusion — Training")
    print("=" * 60)

    loader = MasterDataLoader()
    all_pairs = loader.get_cow_date_pairs()
    if not all_pairs:
        print("[ERROR] No label files. Check LABEL_ROOT in app/ml/sensor_configs.py")
        sys.exit(1)

    sampled_pairs = resolve_sessions(all_pairs, max_sessions, random_state)
    print(f"[1/7] Sessions: {len(sampled_pairs)} / {len(all_pairs)} available")

    print("[2/7] Loading & fusing sensors (may take several minutes)...")
    df = loader._align_pairs(sampled_pairs)
    if df.empty:
        print("[ERROR] No fused data.")
        sys.exit(1)
    print(f"      Rows: {df.shape[0]:,}  Columns: {df.shape[1]}")

    if merge_weak:
        df = merge_weak_classes(df)

    df = df[df["behavior"] >= 0].copy()
    meta_cols = {
        "timestamp", "behavior", "datetime", "label_idx", "sensor_idx",
        "_cow_id", "_date",
    }
    feature_names = [c for c in df.columns if c not in meta_cols]

    y_raw = df["behavior"].values
    y, id_remap = recode_labels_consecutive(y_raw)
    df["behavior"] = y

    # Save human-readable label map (original id → new id)
    inv_encoder = {str(int(k)): int(v) for k, v in id_remap.items()}
    with open(ENCODER_PATH, "w", encoding="utf-8") as f:
        json.dump(inv_encoder, f, indent=2)
    print(f"[3/7] Classes after merge/recode: {sorted(np.unique(y))}")

    X = df[feature_names].copy()
    classes, counts = np.unique(y, return_counts=True)
    print(f"      Class counts: {dict(zip(classes.astype(int), counts.astype(int)))}")

    if len(classes) < 2:
        print("[ERROR] Need at least 2 behavior classes.")
        sys.exit(1)

    print("[4/7] Session-aware train/test split...")
    train_mask, test_mask = session_train_test_split(df, test_size, random_state)
    X_train, y_train = X[train_mask].copy(), y[train_mask]
    X_test, y_test = X[test_mask].copy(), y[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        print("      [Warn] Falling back to random row split.")
        from sklearn.model_selection import train_test_split as tts
        X_train, X_test, y_train, y_test = tts(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    sample_weights = compute_sample_weights(y_train)

    xgb_device = xgboost_device(use_gpu=xgb_gpu)
    if xgb_device == "cuda":
        print(f"[5/7] XGBoost on CUDA — {torch.cuda.get_device_name(0)}")
    else:
        print("[5/7] XGBoost on CPU (recommended: matches pandas data & API inference)")
        if xgb_gpu and not torch.cuda.is_available():
            print("      [Note] --xgb-gpu requested but CUDA not available.")

    # Pass 1: all features → pick top-K
    prelim_params = dict(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.08,
        tree_method="hist",
        device=xgb_device,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    prelim = xgb.XGBClassifier(**prelim_params)
    prelim.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    if top_features > 0 and top_features < len(feature_names):
        selected = top_features_by_importance(
            prelim.feature_importances_, feature_names, top_k=top_features
        )
        X_train = X_train[selected]
        X_test = X_test[selected]
        feature_names = selected
        with open(SELECTED_FEATURES_PATH, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
    else:
        selected = feature_names

    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    print("[6/7] Final training on selected features...")
    default_params = dict(
        n_estimators=400,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=3,
        gamma=0.1,
        tree_method="hist",
        device=xgb_device,
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
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print("[7/7] Evaluation...")
    y_pred = xgboost_predict(model, X_test)
    label_names = [f"class_{c}" for c in sorted(classes)]
    metrics = save_metrics_report(
        y_test,
        y_pred,
        REPORT_PATH,
        CONFUSION_PATH,
        "Master Sensor Fusion Model",
        label_names=label_names,
        extra_lines=[
            f"Sessions used      : {len(sampled_pairs)}",
            f"Features used      : {len(feature_names)}",
            f"Weak classes merged: {merge_weak} → bucket id {MERGED_CLASS_ID}",
            f"Top features cap   : {top_features}",
        ],
    )

    _save_feature_importance(model, feature_names)
    joblib.dump(model, MODEL_PATH)
    print(f"  Model saved → {MODEL_PATH}")

    imp_df = _make_importance_df(model, feature_names)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n\nTop 20 features:\n")
        f.write(imp_df.head(20).to_string())

    print(f"\n  Total time: {time.time() - t0:.1f}s")
    _robustness_check(model, X_test, y_test, feature_names)
    print("\n[Done] Master model ready for the API.\n")
    return metrics


def _save_feature_importance(model, feature_names, top_n=25):
    imp_df = _make_importance_df(model, feature_names).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(imp_df["feature"], imp_df["importance"])
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Features — Master Model")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close()
    print(f"  Feature plot → {IMPORTANCE_PATH}")


def _robustness_check(model, X_test, y_test, feature_names):
    print("\n--- UWB outage test ---")
    uwb_cols = [c for c in feature_names if c.startswith("uwb_")]
    if not uwb_cols:
        return
    X_no = X_test.copy()
    X_no[uwb_cols] = np.nan
    try:
        y_p = xgboost_predict(model, X_no)
        print(f"  F1 macro (no UWB): {f1_score(y_test, y_p, average='macro', zero_division=0):.4f}")
    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Master Fusion model.")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="0 = use ALL sessions. Default: 0")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-features", type=int, default=50,
                        help="Keep top N features. 0 = use all.")
    parser.add_argument("--no-merge-weak", action="store_true",
                        help="Do not merge rare classes 3,5,7")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--xgb-gpu",
        action="store_true",
        help="Train XGBoost on CUDA (default: CPU, avoids device mismatch warnings)",
    )
    args = parser.parse_args()

    train_master_model(
        max_sessions=args.max_sessions,
        test_size=args.test_size,
        random_state=args.seed,
        top_features=args.top_features,
        merge_weak=not args.no_merge_weak,
        xgb_gpu=args.xgb_gpu,
        xgb_params={"n_estimators": args.n_estimators, "max_depth": args.max_depth},
    )
