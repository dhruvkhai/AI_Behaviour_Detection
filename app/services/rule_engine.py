from typing import List, Dict
from app.models.db_models import Alert

class RuleEngine:
    @staticmethod
    def evaluate(sensor_data, feature_df, anomaly_score: int) -> List[Dict]:
        """
        Runs classical rules on top of AI predictions.
        Returns a list of alert details if any rules are triggered.
        """
        alerts = []
        
        # 1. Temperature Check (PT100)
        # Normal cow temp is 38.5 - 39.3 C
        if sensor_data.temperature > 39.5:
            alerts.append({
                "type": "HIGH_TEMP",
                "severity": "HIGH",
                "message": f"Cow fever detected: {sensor_data.temperature}C"
            })
        elif sensor_data.temperature < 37.5:
            alerts.append({
                "type": "LOW_TEMP",
                "severity": "MEDIUM",
                "message": f"Low body temperature: {sensor_data.temperature}C"
            })

        # 2. Activity Check (IMU energy)
        activity = feature_df["activity_intensity"].iloc[0]
        if activity < 0.05: # Threshold determined by prototype testing
            alerts.append({
                "type": "LOW_ACTIVITY",
                "severity": "MEDIUM",
                "message": "Prolonged inactivity / possible lethargy"
            })

        # 3. AI Anomaly Check
        if anomaly_score == -1:
            alerts.append({
                "type": "AI_ANOMALY",
                "severity": "LOW",
                "message": "AI detected unusual behavior pattern"
            })
            
        return alerts
