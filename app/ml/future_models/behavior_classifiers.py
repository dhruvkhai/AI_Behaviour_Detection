"""
TUTORIAL FOR FACULTY PRESENTATION:
This module contains placeholders for the supervised learning phase.
Once we collect enough labeled data (e.g., 'Walking', 'Eating', 'Ruminating'), 
we can swap the Isolation Forest with these models.
"""

class BehaviorClassifier:
    """
    Base class for future behavior classification.
    """
    def predict(self, window_data):
        # TODO: Implement prediction logic
        return "Normal"

class XGBBehaviorModel(BehaviorClassifier):
    """
    XGBoost implementation for tabular feature-based classification.
    Best for: fast, interpretable results from extracted features.
    """
    def train(self, X_train, y_train):
        # TODO: Implement XGBoost training
        pass

class CNN_BiLSTM_Model(BehaviorClassifier):
    """
    Deep Learning implementation for raw time-series classification.
    Best for: High accuracy by learning patterns directly from IMU data.
    """
    def build_model(self):
        # future architecture: 1D CNN for spatial features + BiLSTM for temporal sequence
        pass
