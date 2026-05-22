import json
import os

import joblib
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

class XGBoostClassifier:
    """
    XGBoost implementation for classification based on statistical features.
    """
    def __init__(self, model_path: str = "models/xgboost_model.json"):
        self.model_path = model_path
        self.model = xgb.XGBClassifier()
        if os.path.exists(self.model_path):
            self.model.load_model(self.model_path)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class MasterFusionClassifier:
    """
    XGBoost classifier for Master Sensor Fusion, loaded from joblib.
    """
    def __init__(self, model_path: str = "models/behavior_master.joblib"):
        self.model_path = model_path
        self.model = None
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"Master Model loaded from {self.model_path}")

    def predict(self, X):
        if self.model is None:
            raise ValueError("Master Model not loaded. Train it first.")
        import pandas as pd
        from app.ml.training_utils import xgboost_predict

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.atleast_2d(X))
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return xgboost_predict(self.model, X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Master Model not loaded.")
        import pandas as pd
        from app.ml.training_utils import xgboost_predict_proba

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.atleast_2d(X))
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return xgboost_predict_proba(self.model, X)

class CNNBiLSTMModel(nn.Module):
    """
    PyTorch Implementation of 1D CNN + BiLSTM for time-series sensor data.
    Input Shape: (batch_size, channels, sequence_length)
    """
    def __init__(self, n_channels=86, n_classes=8, hidden_dim=64, n_layers=2):
        super(CNNBiLSTMModel, self).__init__()
        
        # 1. Convolutional Layer for spatial feature extraction
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 2. BiLSTM Layer for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Fully Connected Layer
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last hidden state 
        x = lstm_out[:, -1, :] 
        x = self.dropout(x)
        x = self.fc(x)
        return x

class DeepBehaviorClassifier:
    """
    Wrapper for the CNN-BiLSTM PyTorch model (models/behavior_deep.pth).
    """
    def __init__(
        self,
        model_path: str = "models/behavior_deep.pth",
        meta_path: str = "models/behavior_deep_meta.json",
        n_channels: int = 86,
        n_classes: int = 8,
    ):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._feat_mean = None
        self._feat_std = None
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            n_channels = meta.get("n_channels", n_channels)
            n_classes = meta.get("n_classes", n_classes)
            if "feature_mean" in meta and "feature_std" in meta:
                self._feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
                self._feat_std = np.array(meta["feature_std"], dtype=np.float32)
        self.model = CNNBiLSTMModel(n_channels=n_channels, n_classes=n_classes).to(self.device)
        self._loaded = False
        if os.path.exists(self.model_path):
            try:
                state = torch.load(
                    self.model_path, map_location=self.device, weights_only=True
                )
            except TypeError:
                state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self._loaded = True

    def _normalize(self, x_np: np.ndarray) -> np.ndarray:
        if self._feat_mean is None or self._feat_std is None:
            return x_np
        x = np.asarray(x_np, dtype=np.float32)
        mean = self._feat_mean.reshape(-1, 1)
        std = self._feat_std.reshape(-1, 1)
        return (x - mean) / std

    def predict(self, x_np):
        if not self._loaded:
            raise ValueError("Deep model not trained. Run: python train_dl_model.py")
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(self._normalize(x_np)).to(self.device)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, x_np):
        if not self._loaded:
            raise ValueError("Deep model not trained.")
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(self._normalize(x_np)).to(self.device)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            logits = self.model(x_tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()
