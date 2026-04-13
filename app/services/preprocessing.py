import pandas as pd
from typing import List
from app.models.db_models import SensorData
from app.core.config import settings

class Preprocessor:
    @staticmethod
    def get_window_from_db(db_session, device_id: str, window_size: int = settings.WINDOW_SIZE):
        """
        Retrieves the last n records for a device from the DB and prepares them for feature extraction.
        """
        records = (
            db_session.query(SensorData)
            .filter(SensorData.device_id == device_id)
            .order_by(SensorData.timestamp.desc())
            .limit(window_size)
            .all()
        )
        
        if not records or len(records) < window_size:
            return None
            
        # Reverse to get chronological order
        records = records[::-1]
        
        # Flatten the JSON data for analysis
        data = []
        for r in records:
            payload = r.raw_payload
            data.append({
                "acc_x": payload["imu"]["accel"][0],
                "acc_y": payload["imu"]["accel"][1],
                "acc_z": payload["imu"]["accel"][2],
                "gyro_x": payload["imu"]["gyro"][0],
                "gyro_y": payload["imu"]["gyro"][1],
                "gyro_z": payload["imu"]["gyro"][2],
                "temp": r.temperature,
                "rumination": r.rumination_score
            })
            
        return pd.DataFrame(data)
