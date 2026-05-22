"""
train_dl_model.py
=================
Train the 1D CNN + BiLSTM deep model (backup when master model fails).

Recommended:
  python train_dl_model.py --max-sessions 50 --epochs 30 --batch-size 16
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.ml.fusion_pipeline import MasterDataLoader
from app.ml.classifiers import CNNBiLSTMModel
from app.ml.training_utils import (
    merge_weak_classes,
    normalize_train_test,
    recode_labels_consecutive,
    resolve_sessions,
    sanitize_feature_matrix,
    save_metrics_report,
    session_train_test_split,
    torch_class_weights,
)

SEQ_LEN = 45
EPOCHS = 30
LEARNING_RATE = 3e-4
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "behavior_deep.pth")
META_PATH = os.path.join(MODELS_DIR, "behavior_deep_meta.json")
REPORT_PATH = os.path.join(MODELS_DIR, "report_deep.txt")
CONFUSION_PATH = os.path.join(MODELS_DIR, "confusion_matrix_deep.png")


def resolve_training_device(batch_size_override: int | None = None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        vram_gb = props.total_memory / (1024 ** 3)
        if batch_size_override is not None:
            batch_size = batch_size_override
        elif vram_gb <= 4.5:
            batch_size = 16
        elif vram_gb <= 8:
            batch_size = 32
        else:
            batch_size = 64
    else:
        device = torch.device("cpu")
        gpu_name, vram_gb = "CPU", 0.0
        batch_size = batch_size_override or 32
    return device, gpu_name, vram_gb, batch_size


class CowSequenceDataset(Dataset):
    def __init__(self, X_df, y_array, seq_len=45):
        self.X = X_df.values
        self.y = y_array
        self.seq_len = seq_len
        self.samples = max(0, len(self.X) - seq_len)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        block_x = self.X[idx : idx + self.seq_len]
        block_y = self.y[idx : idx + self.seq_len]
        x_tensor = torch.tensor(block_x.T, dtype=torch.float32)
        vals, counts = np.unique(block_y, return_counts=True)
        y_label = vals[np.argmax(counts)]
        return x_tensor, torch.tensor(y_label, dtype=torch.long)


def run_pipeline(
    max_sessions: int = 50,
    batch_size: int | None = None,
    epochs: int = EPOCHS,
    merge_weak: bool = True,
):
    os.makedirs(MODELS_DIR, exist_ok=True)
    device, gpu_name, vram_gb, batch_size = resolve_training_device(batch_size)
    # AMP can overflow on wide, unscaled sensor features — keep full float32 for stability
    use_amp = False

    print("============================================================")
    print("  Deep Learning (CNN + BiLSTM)")
    if device.type == "cuda":
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB)  batch={batch_size}  float32 (stable)")
    else:
        print("  GPU: not available — using CPU")
    print("============================================================")

    loader = MasterDataLoader()
    all_pairs = loader.get_cow_date_pairs()
    if not all_pairs:
        print("[ERROR] No data.")
        return

    sampled = resolve_sessions(all_pairs, max_sessions, random_state=42)
    print(f"[1/6] Sessions: {len(sampled)}")
    df = loader._align_pairs(sampled)
    if merge_weak:
        df = merge_weak_classes(df)
    df = df[df["behavior"] >= 0].copy()

    meta_cols = {"timestamp", "behavior", "datetime", "label_idx", "sensor_idx", "_cow_id", "_date"}
    feature_names = [c for c in df.columns if c not in meta_cols]

    y_raw = df["behavior"].values
    y, _ = recode_labels_consecutive(y_raw)
    df["behavior"] = y
    n_classes = len(np.unique(y))

    # Use same feature list as master if it exists
    feat_path = os.path.join(MODELS_DIR, "behavior_master_features.json")
    if os.path.exists(feat_path):
        with open(feat_path, encoding="utf-8") as f:
            master_feats = json.load(f)
        feature_names = [c for c in master_feats if c in feature_names]
        print(f"      Using {len(feature_names)} features aligned with master model")

    X = sanitize_feature_matrix(df[feature_names])
    print(f"      Features: {len(feature_names)} (sanitized inf/outliers)")

    print("[2/6] Session-aware split...")
    train_mask, test_mask = session_train_test_split(df, test_size=0.2, random_state=42)
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    y_train = y[train_mask]
    y_test = y[test_mask]

    X_train, X_test, feat_mean, feat_std = normalize_train_test(X_train, X_test)
    print("      Applied z-score normalization (train stats)")

    class_weights = torch_class_weights(y_train, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print(f"[3/6] Building {SEQ_LEN}s sequences...")
    train_ds = CowSequenceDataset(X_train, y_train, SEQ_LEN)
    test_ds = CowSequenceDataset(X_test, y_test, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=use_amp)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=use_amp)
    print(f"      Train windows: {len(train_ds)} | Test: {len(test_ds)}")

    print("[4/6] Training...")
    model = CNNBiLSTMModel(n_channels=len(feature_names), n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, tracked, n_batches = 0.0, 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if not torch.isfinite(loss):
                print("      [ERROR] Loss is NaN — check data or lower learning rate.")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            _, pred = torch.max(outputs, 1)
            tracked += labels.size(0)
            correct += (pred == labels).sum().item()
        avg_loss = total_loss / max(n_batches, 1)
        print(
            f"      Epoch {epoch+1:02d}/{epochs} | "
            f"loss={avg_loss:.4f} | train_acc={100*correct/tracked:.1f}%"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    meta = {
        "n_channels": len(feature_names),
        "n_classes": n_classes,
        "seq_len": SEQ_LEN,
        "feature_names": feature_names,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"      Weights → {MODEL_PATH}")

    print("[5/6] Test evaluation...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            preds.extend(pred.cpu().numpy())
            targets.extend(labels.numpy())

    label_names = [f"class_{i}" for i in range(n_classes)]
    save_metrics_report(
        np.array(targets),
        np.array(preds),
        REPORT_PATH,
        CONFUSION_PATH,
        "Deep CNN-BiLSTM Model",
        label_names=label_names,
        extra_lines=[
            f"Sessions : {len(sampled)}",
            f"Epochs   : {epochs}",
            f"Features : {len(feature_names)}",
        ],
    )
    print(f"[6/6] Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-sessions", type=int, default=50, help="0 = all sessions")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--no-merge-weak", action="store_true")
    args = p.parse_args()
    run_pipeline(
        max_sessions=args.max_sessions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        merge_weak=not args.no_merge_weak,
    )
