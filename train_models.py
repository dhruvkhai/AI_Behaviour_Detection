import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.ml.anomaly import AnomalousBehaviorDetector
from app.ml.classifiers import XGBoostClassifier, DeepBehaviorClassifier
from scipy.stats import skew, kurtosis

def extract_statistical_features(window):
    """
    Extracts features for XGBoost and Isolation Forest.
    """
    # Assuming window shape is (channels, seq_len)
    feats = []
    for channel in window:
        feats.extend([
            np.mean(channel),
            np.std(channel),
            np.min(channel),
            np.max(channel),
            skew(channel),
            kurtosis(channel),
            np.sqrt(np.mean(channel**2)) # RMS
        ])
    return np.array(feats)

def prepare_data(data_path, window_size=100, step_size=50):
    """
    Loads raw data and prepares sliding windows.
    Mock implementation for demonstration.
    """
    print(f"Loading data from {data_path}...")
    # In a real scenario, use pd.read_csv with chunks for 3.8GB files
    # df = pd.read_csv(data_path) 
    
    # Dummy data for structure demonstration
    dummy_len = 10000
    dummy_data = np.random.randn(dummy_len, 3) # x, y, z acceleration
    
    windows = []
    stat_features = []
    
    for i in range(0, dummy_len - window_size, step_size):
        window = dummy_data[i:i+window_size].T # (3, 100)
        windows.append(window)
        stat_features.append(extract_statistical_features(window))
        
    return np.array(windows), np.array(stat_features)

def main():
    # 1. Prepare Data
    # Note: User should provide actual path to extracted sensor data
    data_path = "sensor_data_extracted/raw_imu.csv" 
    X_raw, X_stats = prepare_data(data_path)
    
    # 2. Train Anomaly Detection (Priority 1)
    print("Training Anomaly Detector (Isolation Forest)...")
    anomaly_detector = AnomalousBehaviorDetector()
    anomaly_detector.train(pd.DataFrame(X_stats))
    print("Anomaly Detector saved.")

    # 3. Train XGBoost (Classical Baseline)
    # y_train = np.random.randint(0, 5, size=len(X_stats)) # Mock labels
    # print("Training XGBoost Classifier...")
    # xgb_clf = XGBoostClassifier()
    # xgb_clf.train(X_stats, y_train)
    # print("XGBoost Model saved.")

    # 4. Train Deep Learning (1D CNN + BiLSTM)
    print("Training Deep Learning Model (CNN + BiLSTM)...")
    # Convert to Tensors
    # X_raw_tensor = torch.FloatTensor(X_raw)
    # y_tensor = torch.LongTensor(y_train)
    # dataset = TensorDataset(X_raw_tensor, y_tensor)
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    deep_clf = DeepBehaviorClassifier(n_channels=3, n_classes=5)
    # deep_clf.train(loader, epochs=5)
    print("Deep Learning Model architecture ready (Training loop commented out for safety).")

if __name__ == "__main__":
    main()
