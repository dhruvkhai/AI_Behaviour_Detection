import torch
import torch.nn as nn
import xgboost as xgb
import os
import joblib
import numpy as np

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

class CNNBiLSTMModel(nn.Module):
    """
    PyTorch Implementation of 1D CNN + BiLSTM for time-series sensor data.
    Input Shape: (batch_size, channels, sequence_length)
    """
    def __init__(self, n_channels=3, n_classes=5, hidden_dim=64, n_layers=2):
        super(CNNBiLSTMModel, self).__init__()
        
        # 1. Convolutional Layer for spatial feature extraction
        self.conv1 = nn.Conv1d(in_channels=n_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # 2. BiLSTM Layer for temporal dependencies
        # After max pool, seq_len is halved. input_size must match out_channels of Conv1d
        self.lstm = nn.LSTM(
            input_size=32, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 3. Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, n_classes) # * 2 because bidirectional

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
        
        # Take the last hidden state for classification
        # We can also use global average pooling here
        x = lstm_out[:, -1, :] 
        
        x = self.fc(x)
        return x

class DeepBehaviorClassifier:
    """
    Wrapper for the CNN-BiLSTM PyTorch model.
    """
    def __init__(self, model_path: str = "models/deep_model.pth", n_channels=3, n_classes=5):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNBiLSTMModel(n_channels=n_channels, n_classes=n_classes).to(self.device)
        
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
    def predict(self, x_np):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_np).to(self.device)
            # Ensure shape is (batch, channels, seq_len)
            if len(x_tensor.shape) == 2:
                x_tensor = x_tensor.unsqueeze(0)
            
            outputs = self.model(x_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def train(self, train_loader, epochs=10, lr=0.001):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.model_path)
