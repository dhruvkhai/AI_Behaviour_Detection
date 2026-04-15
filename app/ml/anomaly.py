import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AnomalousBehaviorDetector:
    """
    Isolation Forest implementation for detecting unusual behavior patterns.
    Useful for health monitoring where 'healthy' behavior is common and 
    'unhealthy' behavior is an anomaly.
    """
    def __init__(self, model_path: str = "models/isolation_forest.joblib"):
        self.model_path = model_path
        self.model = None
        self.contamination = 0.05  # Assume 5% of data is anomalous by default
        
        if os.path.exists(self.model_path):
            self.load_model()

    def train(self, features: pd.DataFrame):
        """
        Trains the Isolation Forest model on provided features.
        Features should be statistical summaries of sliding windows.
        """
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(features)
        self.save_model()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Returns -1 for anomalies, 1 for normal data.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.predict(features)

    def score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """
        Returns raw anomaly scores (lower is more anomalous).
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        return self.model.score_samples(features)

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        self.model = joblib.load(self.model_path)
