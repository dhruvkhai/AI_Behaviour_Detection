# Sensor configurations for the AI Behaviour Detection project
import os

DATA_ROOT = r"C:\Users\User\sensor_data"

SENSOR_CONFIGS = {
    "immu": {
        "sub_path": os.path.join("main_data", "immu"),
        "id_prefix": "T",
        "columns": ["timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "mag_x_uT", "mag_y_uT", "mag_z_uT"],
        "feature_cols": ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2", "mag_x_uT", "mag_y_uT", "mag_z_uT"],
        "freq": 10, # Hz
        "window_size": 10 # seconds
    },
    "cbt": {
        "sub_path": os.path.join("main_data", "cbt"),
        "id_prefix": "C",
        "columns": ["timestamp", "temperature_C"],
        "feature_cols": ["temperature_C"],
        "freq": 1/60, # Hz (1 sample per min)
        "window_size": 300 # seconds (5 mins)
    },
    "pressure": {
        "sub_path": os.path.join("main_data", "pressure"),
        "id_prefix": "T",
        "columns": ["timestamp", "pressure_Pa", "elevation_m"],
        "feature_cols": ["pressure_Pa", "elevation_m"],
        "freq": 10, # Hz
        "window_size": 10
    },
    "milk": {
        "sub_path": os.path.join("main_data", "milk"),
        "id_prefix": "C",
        "columns": ["timestamp", "milk_weight_kg", "DIM"],
        "feature_cols": ["milk_weight_kg", "DIM"],
        "freq": 1/86400, # Daily
        "window_size": 86400
    },
    "thi": {
        "sub_path": os.path.join("main_data", "thi"),
        "is_global": True, # Applied to all cows
        "file_name": "average.csv",
        "columns": ["timestamp", "temperature_F", "humidity_per", "THI"],
        "feature_cols": ["temperature_F", "humidity_per", "THI"],
        "freq": 1/60, # 1 min
        "window_size": 300
    },
    "uwb": {
        "sub_path": os.path.join("main_data", "uwb"),
        "id_prefix": "T",
        "columns": ["timestamp", "coord_x_cm", "coord_y_cm", "coord_z_cm"],
        "feature_cols": ["coord_x_cm", "coord_y_cm", "coord_z_cm"],
        "freq": 1/15, # Sample every 15s
        "window_size": 60
    },
    "weather": {
        "sub_path": os.path.join("main_data", "weather"),
        "is_global": True,
        "is_excel": True,
        "freq": 1/3600, # Assuming hourly
        "window_size": 3600
    }
}

LABEL_ROOT = os.path.join(DATA_ROOT, "behavior_labels", "individual")
