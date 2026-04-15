from app.models.db_models import Alert
from datetime import datetime

class AlertEngine:
    """
    Translates model predictions and scores into actionable alerts.
    """
    
    def __init__(self, high_anomaly_threshold=-0.8, med_anomaly_threshold=-0.6):
        self.high_threshold = high_anomaly_threshold
        self.med_threshold = med_anomaly_threshold

    def process_anomaly_score(self, device_id: str, score: float) -> Alert:
        """
        Creates an Alert object if the anomaly score cross thresholds.
        """
        if score < self.high_threshold:
            return Alert(
                device_id=device_id,
                alert_type='ANOMALY_CRITICAL',
                severity='HIGH',
                message=f"Critical behavior anomaly detected (score: {score:.2f})",
                timestamp=datetime.now()
            )
        elif score < self.med_threshold:
            return Alert(
                device_id=device_id,
                alert_type='ANOMALY_WARNING',
                severity='MEDIUM',
                message=f"Moderate behavior anomaly detected (score: {score:.2f})",
                timestamp=datetime.now()
            )
        return None

    def process_class_prediction(self, device_id: str, behavior: str) -> Alert:
        """
        Logic for specific behavioral alerts (e.g., Lethargy).
        """
        if behavior == "Lethargic":
            return Alert(
                device_id=device_id,
                alert_type='LOW_ACTIVITY',
                severity='MEDIUM',
                message="Cow exhibiting low activity levels compared to baseline.",
                timestamp=datetime.now()
            )
        # Add more rules as needed
        return None
