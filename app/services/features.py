import pandas as pd
import numpy as np

class FeatureExtractor:
    @staticmethod
    def compute_features(df: pd.DataFrame):
        """
        Converts a window of raw sensor data into a single feature vector for AI models.
        """
        features = {}
        
        # Motion Features (IMU)
        for col in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_energy"] = np.sum(df[col]**2) / len(df)
            
        # Overall activity level (Magnitude of Acceleration)
        acc_mag = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        features["activity_intensity"] = acc_mag.mean()
        
        # Temperature Features
        features["temp_avg"] = df["temp"].mean()
        features["temp_max"] = df["temp"].max()
        
        # Rumination Features
        features["rumination_avg"] = df["rumination"].mean()
        
        return pd.DataFrame([features])
