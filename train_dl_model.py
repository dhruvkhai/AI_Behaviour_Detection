"""
train_dl_model.py
=================
Deep Learning Training Pipeline (1D CNN + BiLSTM)

Converts a time-series Pandas DataFrame of sensor telemetry into 45-second overlapping sequence tensors
and trains a PyTorch sequence classifier using hardware acceleration.
"""

import os
import sys
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Adjust paths if executed directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.ml.fusion_pipeline import MasterDataLoader
from app.ml.classifiers import CNNBiLSTMModel

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------
SEQ_LEN = 45          # 45-second sliding windows
BATCH_SIZE = 256      # Memory-efficient mini-batches
EPOCHS = 15           # Deep learning sweeping epochs
LEARNING_RATE = 0.001 # Convergence step

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "behavior_deep.pth")

# ---------------------------------------------------------------------------
#  Dataset Class
# ---------------------------------------------------------------------------
class CowSequenceDataset(Dataset):
    """
    Slices raw (Rows, Features) matrices into PyTorch sequence tensors:
    (Channels, Sequence_Length).
    """
    def __init__(self, X_df, y_array, seq_len=45):
        self.X = X_df.values
        self.y = y_array
        self.seq_len = seq_len
        # Generate indices for continuous sliding windows natively 
        self.samples = max(0, len(self.X) - seq_len)

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # Slice the window block
        block_x = self.X[idx : idx + self.seq_len]
        block_y = self.y[idx : idx + self.seq_len]
        
        # Convert X to (Channels, Sequence_Length)
        x_tensor = torch.tensor(block_x.T, dtype=torch.float32)
        
        # Ground Truth Label for the sequence = The mathematical Mode of the window
        vals, counts = np.unique(block_y, return_counts=True)
        y_label = vals[np.argmax(counts)]
        y_tensor = torch.tensor(y_label, dtype=torch.long)
        
        return x_tensor, y_tensor


def run_pipeline():
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("============================================================")
    print("  Deep Learning Sensor AI — Training Pipeline (PyTorch)")
    print(f"  Target Hardware: {str(device).upper()}")
    print("============================================================")

    # 1. Discover sessions & Load
    loader = MasterDataLoader()
    all_pairs = loader.get_cow_date_pairs()
    
    if not all_pairs:
        print("[ERROR] No data found.")
        return
        
    print(f"[1/5] Loading data from {len(all_pairs)} tracking sessions...")
    sampled_pairs = loader.stratified_sample(all_pairs, max_sessions=15, random_state=42)
    df = loader._align_pairs(sampled_pairs)
    
    # Prune unknown classes
    df = df[df["behavior"] >= 0].copy()
    
    meta_cols = {"timestamp", "behavior", "datetime", "label_idx", "sensor_idx", "_cow_id", "_date"}
    feature_names = [c for c in df.columns if c not in meta_cols]
    
    # 2. Extract Data
    y = df["behavior"].values
    X = df[feature_names].copy()
    
    # Clean NaNs in PyTorch (XGBoost handled them natively, PyTorch hates them)
    X = X.fillna(0.0)

    # Split (Temporal Split, no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. Build Sequences
    print(f"[2/5] Constructing {SEQ_LEN}-second computational sequences...")
    train_dataset = CowSequenceDataset(X_train, y_train, seq_len=SEQ_LEN)
    test_dataset = CowSequenceDataset(X_test, y_test, seq_len=SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"      Train Arrays: {len(train_dataset)} | Test Arrays: {len(test_dataset)}")
    
    # 4. Neural Network
    print(f"[3/5] Instantiating 1D CNN + BiLSTM Architecture...")
    model = CNNBiLSTMModel(n_channels=len(feature_names), n_classes=len(loader.label_encoder)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"[4/5] Initiating Hardware Backpropagation Loop (Epochs: {EPOCHS})...")
    
    # Training Loop
    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        tracked = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            tracked += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f"      Epoch [{epoch+1:02d}/{EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Acc: {(correct/tracked)*100:.2f}%")
        
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"      State Dictionary saved -> {MODEL_PATH}")
    
    # 5. Verification
    print(f"[5/5] Deep Inference Validation...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_targets, all_preds)
    
    print("\n------------------------------------------------------------")
    print(f"  DL Neural Network Time : {time.time() - t0:.1f}s")
    print(f"  DL General Accuracy    : {acc*100:.2f}%")
    print(f"  DL F1 (Macro) Score    : {f1:.4f}")
    print("------------------------------------------------------------")
    print("Deep Sequence Training Complete.")

if __name__ == "__main__":
    run_pipeline()
