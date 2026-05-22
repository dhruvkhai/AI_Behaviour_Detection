"""
Shared helpers for training scripts (split, class weights, label merge, metrics).
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# Rare / noisy behavior IDs from past reports — merged into one bucket before training.
WEAK_CLASS_IDS = (3, 5, 7)
MERGED_CLASS_ID = 8


def xgboost_device(use_gpu: bool = False) -> str:
    """
    XGBoost device for tabular training.
    Default CPU: data and API inference are pandas on CPU (avoids cuda/cpu mismatch).
    """
    if not use_gpu or xgb is None:
        return "cpu"
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def xgboost_predict(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predict class labels with the same device as the trained booster (no mismatch warning).
    """
    if xgb is None:
        raise ImportError("xgboost is required")

    X_np = np.ascontiguousarray(X.values, dtype=np.float32)
    device = (model.get_params().get("device") or "cpu").lower()

    if device.startswith("cuda"):
        dmat = xgb.QuantileDMatrix(X_np, device="cuda")
        proba = model.get_booster().predict(dmat)
        n_classes = len(model.classes_)
        proba = proba.reshape(-1, n_classes)
        return np.argmax(proba, axis=1).astype(int)

    return model.predict(X_np)


def xgboost_predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities with device aligned to the trained booster."""
    if xgb is None:
        raise ImportError("xgboost is required")

    X_np = np.ascontiguousarray(X.values, dtype=np.float32)
    device = (model.get_params().get("device") or "cpu").lower()

    if device.startswith("cuda"):
        dmat = xgb.QuantileDMatrix(X_np, device="cuda")
        proba = model.get_booster().predict(dmat)
        return proba.reshape(-1, len(model.classes_))

    return model.predict_proba(X_np)


def resolve_sessions(
    all_pairs: list,
    max_sessions: int,
    random_state: int = 42,
) -> list:
    """Use all sessions when max_sessions <= 0, else stratified sample."""
    if max_sessions <= 0 or max_sessions >= len(all_pairs):
        return all_pairs
    from app.ml.fusion_pipeline import MasterDataLoader

    loader = MasterDataLoader()
    return loader.stratified_sample(all_pairs, max_sessions=max_sessions, random_state=random_state)


def merge_weak_classes(df: pd.DataFrame, weak: tuple[int, ...] = WEAK_CLASS_IDS) -> pd.DataFrame:
    """Combine hard-to-learn classes into one 'merged rare' bucket."""
    out = df.copy()
    mask = out["behavior"].isin(weak)
    n = int(mask.sum())
    if n:
        out.loc[mask, "behavior"] = MERGED_CLASS_ID
        print(f"      Merged weak classes {weak} → id {MERGED_CLASS_ID} ({n:,} rows)")
    return out


def recode_labels_consecutive(y: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Map arbitrary class IDs to 0..K-1 for sklearn / PyTorch."""
    unique = sorted(np.unique(y))
    mapping = {old: new for new, old in enumerate(unique)}
    y_new = np.array([mapping[v] for v in y], dtype=np.int64)
    return y_new, mapping


def session_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split by whole (cow, date) sessions — avoids leaking the same day into train and test.
    Returns boolean masks (train_mask, test_mask).
    """
    sessions = df[["_cow_id", "_date"]].drop_duplicates()
    train_sess, test_sess = train_test_split(
        sessions, test_size=test_size, random_state=random_state
    )
    idx = df.set_index(["_cow_id", "_date"]).index
    train_index = pd.MultiIndex.from_frame(train_sess)
    train_mask = np.asarray(idx.isin(train_index), dtype=bool)
    test_mask = ~train_mask
    return train_mask, test_mask


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Balanced weights so rare behaviors matter more during training."""
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_to_w = {int(c): float(w) for c, w in zip(classes, weights)}
    return np.array([class_to_w[int(v)] for v in y], dtype=np.float32)


def sanitize_feature_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/NaN and clip extreme values so neural nets do not overflow."""
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = X.values.astype(np.float32)
    # Per-column clip to robust range (kurtosis / UWB coords can be huge)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if np.all(col == 0):
            continue
        lo, hi = np.nanpercentile(col, [1, 99])
        if hi <= lo:
            hi = lo + 1.0
        arr[:, j] = np.clip(col, lo - 3 * (hi - lo), hi + 3 * (hi - lo))
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def normalize_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Z-score using train statistics only (required for stable deep learning)."""
    mean = X_train.mean(axis=0).values.astype(np.float32)
    std = X_train.std(axis=0).values.astype(np.float32)
    std[std < 1e-6] = 1.0
    X_train_n = (X_train.values - mean) / std
    X_test_n = (X_test.values - mean) / std
    return (
        pd.DataFrame(X_train_n, columns=X_train.columns),
        pd.DataFrame(X_test_n, columns=X_test.columns),
        mean,
        std,
    )


def torch_class_weights(y: np.ndarray, n_classes: int) -> "torch.Tensor":
    """Balanced class weights, clipped to avoid NaN loss with mixed precision."""
    import torch

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    ordered = np.ones(n_classes, dtype=np.float32)
    for c, w in zip(classes, weights):
        ordered[int(c)] = float(w)
    ordered = np.clip(ordered, 0.25, 5.0)
    return torch.tensor(ordered, dtype=torch.float32)


def top_features_by_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_k: int = 50,
) -> list[str]:
    """Keep the top_k most useful sensor columns."""
    order = np.argsort(importances)[::-1]
    k = min(top_k, len(feature_names))
    selected = [feature_names[i] for i in order[:k]]
    print(f"      Feature selection: {len(feature_names)} → {k} columns")
    return selected


def save_metrics_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    report_path: str,
    confusion_path: str,
    title: str,
    label_names: list[str] | None = None,
    extra_lines: list[str] | None = None,
) -> dict:
    """Save text report + confusion matrix image; return summary metrics."""
    classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    if label_names is None:
        label_names = [str(c) for c in classes]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    names = [label_names[i] if i < len(label_names) else f"class_{c}" for i, c in enumerate(classes)]
    report_str = classification_report(
        y_true, y_pred, labels=classes, target_names=names, zero_division=0,
    )

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * 50 + "\n")
        if extra_lines:
            for line in extra_lines:
                f.write(line + "\n")
        f.write(f"Accuracy (overall)     : {acc:.4f}\n")
        f.write(f"F1 macro (fair avg)    : {f1_macro:.4f}\n")
        f.write(f"F1 weighted            : {f1_weighted:.4f}\n\n")
        f.write("Per-class precision / recall / F1:\n")
        f.write(report_str)

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=150)
    plt.close()

    print(f"\n  Accuracy     : {acc:.4f}")
    print(f"  F1 (macro)   : {f1_macro:.4f}  ← main score (treats all classes equally)")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(report_str)
    print(f"  Report → {report_path}")
    print(f"  Confusion matrix → {confusion_path}")

    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def ensemble_predict(
    master_proba: np.ndarray,
    deep_proba: np.ndarray,
    master_weight: float = 0.65,
) -> np.ndarray:
    """Weighted average of probability vectors (same number of classes)."""
    if master_proba.shape != deep_proba.shape:
        n = min(master_proba.shape[1], deep_proba.shape[1])
        master_proba = master_proba[:, :n]
        deep_proba = deep_proba[:, :n]
    combined = master_weight * master_proba + (1.0 - master_weight) * deep_proba
    return np.argmax(combined, axis=1)
