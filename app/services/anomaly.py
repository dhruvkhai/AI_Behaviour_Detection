from sklearn.ensemble import IsolationForest
import joblib
import os
from app.core.config import settings

MODEL_PATH = "app/ml/artifacts/isolation_forest.joblib"

class AnomalyDetector:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            return joblib.load(MODEL_PATH)
        else:
            # Initialize a new model if none exists (Prototype mode)
            # In production, you would train this on historical normal data
            return IsolationForest(contamination=0.1, random_state=42)

    def predict(self, feature_df):
        """
        Returns -1 for anomaly, 1 for normal.
        """
        try:
            # If model isn't trained, we can't predict much
            # For the prototype, we assume it's "partially trained" or using a generic fit
            # Here we just return a placeholder score if not fitted
            prediction = self.model.predict(feature_df)
            return int(prediction[0])
        except Exception:
            # Fallback for unfitted model in first-run prototype
            return 1 

    def train(self, feature_df):
        """
        Fits the model on new data.
        """
        self.model.fit(feature_df)
        joblib.dump(self.model, MODEL_PATH)
