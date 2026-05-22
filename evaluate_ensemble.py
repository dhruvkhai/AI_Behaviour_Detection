"""
evaluate_ensemble.py
====================
Combine Master XGBoost + Deep CNN predictions (offline check).

Run after both models are trained:
  python evaluate_ensemble.py
"""

import os
import json
import sys

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.ml.classifiers import CNNBiLSTMModel
from app.ml.fusion_pipeline import MasterDataLoader
from app.ml.training_utils import (
    ensemble_predict,
    merge_weak_classes,
    recode_labels_consecutive,
    resolve_sessions,
    save_metrics_report,
    session_train_test_split,
    xgboost_predict_proba,
)
from train_dl_model import CowSequenceDataset, SEQ_LEN

MODELS_DIR = "models"
MASTER_PATH = os.path.join(MODELS_DIR, "behavior_master.joblib")
DEEP_PATH = os.path.join(MODELS_DIR, "behavior_deep.pth")
META_PATH = os.path.join(MODELS_DIR, "behavior_deep_meta.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "behavior_master_features.json")
REPORT_PATH = os.path.join(MODELS_DIR, "report_ensemble.txt")
CONFUSION_PATH = os.path.join(MODELS_DIR, "confusion_matrix_ensemble.png")


def main(max_sessions: int = 0, master_weight: float = 0.65):
    if not os.path.exists(MASTER_PATH) or not os.path.exists(DEEP_PATH):
        print("Train both models first:")
        print("  python train_master_model.py --max-sessions 0")
        print("  python train_dl_model.py --max-sessions 50 --epochs 30")
        return

    with open(FEATURES_PATH, encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    loader = MasterDataLoader()
    pairs = resolve_sessions(loader.get_cow_date_pairs(), max_sessions, 42)
    df = loader._align_pairs(pairs)
    df = merge_weak_classes(df)
    df = df[df["behavior"] >= 0].copy()
    y_raw = df["behavior"].values
    y, _ = recode_labels_consecutive(y_raw)
    df["behavior"] = y

    meta_cols = {"timestamp", "behavior", "datetime", "label_idx", "sensor_idx", "_cow_id", "_date"}
    all_feats = [c for c in df.columns if c not in meta_cols]
    X = df[[c for c in feature_names if c in all_feats]].fillna(0.0)
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan

    _, test_mask = session_train_test_split(df, 0.2, 42)
    X_test = X[test_mask].reset_index(drop=True)
    y_test = y[test_mask]

    master = joblib.load(MASTER_PATH)
    master_proba = xgboost_predict_proba(master, X_test[feature_names])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTMModel(
        n_channels=meta["n_channels"],
        n_classes=meta["n_classes"],
    ).to(device)
    model.load_state_dict(torch.load(DEEP_PATH, map_location=device))
    model.eval()

    dl_feats = [c for c in meta["feature_names"] if c in X_test.columns]
    X_dl = X_test[dl_feats].fillna(0.0)
    ds = CowSequenceDataset(X_dl, y_test, SEQ_LEN)
    deep_proba = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)
            logits = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
            deep_proba.append(prob)
    deep_proba = np.array(deep_proba)
    y_test_seq = np.array([ds[i][1].item() for i in range(len(ds))])
    master_proba = master_proba[SEQ_LEN: SEQ_LEN + len(deep_proba)]

    y_pred = ensemble_predict(master_proba, deep_proba, master_weight)
    label_names = [f"class_{i}" for i in range(meta["n_classes"])]

    save_metrics_report(
        y_test_seq,
        y_pred,
        REPORT_PATH,
        CONFUSION_PATH,
        f"Ensemble (master {master_weight:.0%} + deep)",
        label_names=label_names,
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--master-weight", type=float, default=0.65)
    a = ap.parse_args()
    main(a.max_sessions, a.master_weight)
