import numpy as np
import pandas as pd
from app.ml.anomaly import AnomalousBehaviorDetector
from app.ml.classifiers import XGBoostClassifier, DeepBehaviorClassifier
from app.ml.post_processing import PostProcessor
from app.ml.rule_engine import AlertEngine

class DetectionPipeline:
    """
    Unified pipeline to handle multiple models and post-processing.
    """
    def __init__(self):
        self.anomaly_detector = AnomalousBehaviorDetector()
        self.xgb_classifier = XGBoostClassifier()
        self.deep_classifier = DeepBehaviorClassifier()
        self.post_processor = PostProcessor()
        self.alert_engine = AlertEngine()

    def run_inference(self, device_id: str, raw_data: np.ndarray, features: pd.DataFrame):
        """
        Runs the full detection pipeline.
        raw_data: (channels, seq_len) for deep model.
        features: statistical features for XGB and Isolation Forest.
        """
        # 1. Anomaly Detection
        anomaly_scores = self.anomaly_detector.score_samples(features)
        smoothed_scores = self.post_processor.temporal_smoothing(anomaly_scores)
        
        # 2. Classification (Supervised)
        # Using Deep Model as primary classifier
        behavior_ids = self.deep_classifier.predict(raw_data)
        smoothed_behaviors = self.post_processor.majority_voting(behavior_ids.tolist())

        # 3. Alert Generation
        alerts = []
        
        # Check for anomalies
        latest_score = smoothed_scores[-1]
        alert = self.alert_engine.process_anomaly_score(device_id, latest_score)
        if alert:
            alerts.append(alert)
            
        # Check for specific behaviors (mapping IDs to labels would happen here)
        # Placeholder mapping
        id_to_label = {0: "Normal", 1: "Eating", 2: "Ruminating", 3: "Walking", 4: "Lethargic"}
        latest_behavior = id_to_label.get(smoothed_behaviors[-1], "Unknown")
        
        behavior_alert = self.alert_engine.process_class_prediction(device_id, latest_behavior)
        if behavior_alert:
            alerts.append(behavior_alert)

        return {
            "behavior": latest_behavior,
            "anomaly_score": latest_score,
            "alerts": alerts
        }
