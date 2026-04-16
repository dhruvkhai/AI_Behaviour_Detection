import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from scipy.stats import skew, kurtosis
from app.ml.sensor_configs import SENSOR_CONFIGS, LABEL_ROOT, DATA_ROOT

def extract_features(window):
    """
    Extracts statistical features from a sensor window.
    """
    feats = []
    # Columns to exclude from features
    exclude = ['timestamp', 'behavior', 'datetime']
    
    for col in window.columns:
        if col in exclude:
            continue
        try:
            data = pd.to_numeric(window[col], errors='coerce').dropna().values
            if len(data) == 0:
                feats.extend([0, 0, 0, 0, 0, 0])
                continue
            feats.extend([
                np.mean(data),
                np.std(data),
                np.min(data),
                np.max(data),
                skew(data) if len(data) > 2 else 0,
                kurtosis(data) if len(data) > 2 else 0
            ])
        except Exception:
            # If still can't convert, skip
            continue
    return np.array(feats)

def align_and_load(sensor_name, tags=None, limit_run=False):
    """
    Loads data for a specific sensor and aligns it with behavior labels.
    Now supports global features and Excel files.
    """
    config = SENSOR_CONFIGS[sensor_name]
    all_features_dfs = []
    
    label_files = [f for f in os.listdir(LABEL_ROOT) if f.endswith(".csv")]
    
    # If it's a global sensor, we might only need to load one or a few files 
    # and broadcast them to all cows.
    is_global = config.get("is_global", False)
    is_excel = config.get("is_excel", False)

    if limit_run:
        label_files = label_files[:2]
        print(f"LIMIT_RUN active: Only processing {len(label_files)} files.")
    
    print(f"--- Processing Sensor: {sensor_name} ---")
    
    for l_file in label_files:
        cow_id = l_file.split("_")[0]
        date_suffix = l_file.split("_")[1] # e.g. 0725.csv
        date_str = date_suffix.replace(".csv", "") # 0725
        
        sensor_file = None
        if is_global:
            if sensor_name == "weather":
                # Weather files are 07_25_2023.xlsx
                mm = date_str[:2]
                dd = date_str[2:]
                sensor_file = os.path.join(DATA_ROOT, config["sub_path"], f"{mm}_{dd}_2023.xlsx")
            elif "file_name" in config:
                sensor_file = os.path.join(DATA_ROOT, config["sub_path"], config["file_name"])
        else:
            sensor_id = cow_id.replace("C", "T") if config["id_prefix"] == "T" else cow_id
            sensor_folder = os.path.join(DATA_ROOT, config["sub_path"], sensor_id)
            if not os.path.exists(sensor_folder):
                 sensor_file = os.path.join(DATA_ROOT, config["sub_path"], f"{sensor_id}.csv")
            else:
                 sensor_file = os.path.join(sensor_folder, f"{sensor_id}_{date_suffix}")

        if not sensor_file or not os.path.exists(sensor_file):
            continue
            
        print(f"Loading data for {l_file}...")
        labels_df = pd.read_csv(os.path.join(LABEL_ROOT, l_file))
        
        if is_excel:
            sensor_df = pd.read_excel(sensor_file)
        else:
            sensor_df = pd.read_csv(sensor_file)
            
        # Standardize timestamp
        if 'timestamp' not in sensor_df.columns and sensor_name == 'weather':
             # Weather has 'Time' like '12:00 AM'. We need to combine it with current date.
             # date_str is e.g. '0721' (MMDD)
             year = "2023"
             mm = date_str[:2]
             dd = date_str[2:]
             try:
                 # Combine Time column with Date
                 def parse_time(t_str):
                     try:
                         # Handle "12:00 AM" or similar
                         full_str = f"{year}-{mm}-{dd} {t_str}"
                         return pd.to_datetime(full_str).timestamp()
                     except:
                         return np.nan
                 
                 sensor_df['timestamp'] = sensor_df['Time'].apply(parse_time)
             except Exception as e:
                 print(f"Weather time parse error: {e}")
                 sensor_df['timestamp'] = labels_df['timestamp'].iloc[0] # Fallback
        
        # 1. Calculate Rolling Stats on Sensor Data (Vectorized)
        win_samples = int(config["window_size"] * config["freq"])
        if win_samples < 1: win_samples = 1
        
        num_sensor = sensor_df.select_dtypes(include=[np.number]).copy()
        if 'timestamp' in num_sensor.columns:
            rolling = num_sensor.rolling(window=win_samples, center=True)
            
            feats_df = pd.DataFrame({'timestamp': sensor_df['timestamp']})
            for col in num_sensor.columns:
                if col == 'timestamp': continue
                feats_df[f"{col}_mean"] = rolling[col].mean()
                feats_df[f"{col}_std"] = rolling[col].std()
                feats_df[f"{col}_min"] = rolling[col].min()
                feats_df[f"{col}_max"] = rolling[col].max()
                
            # 2. Align labels with the calculated features
            labels_df['timestamp'] = labels_df['timestamp'].astype(float)
            feats_df['timestamp'] = feats_df['timestamp'].astype(float)
            
            aligned = pd.merge_asof(
                labels_df.sort_values('timestamp'),
                feats_df.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
        else:
            # Broadcast mean features if timestamp is missing (e.g. daily weather)
            means = num_sensor.mean()
            for col, val in means.items():
                labels_df[f"{col}_static"] = val
            aligned = labels_df

        all_features_dfs.append(aligned.dropna())
            
    if not all_features_dfs:
        return np.array([]), np.array([])
        
    final_df = pd.concat(all_features_dfs)
    y = final_df['behavior'].values
    drop_cols = ['timestamp', 'behavior', 'datetime', 'label_idx', 'sensor_idx']
    X = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns], errors='ignore').values
    
    return X, y

def train_sensor_model(sensor_name):
    # 1. Load Data
    X, y = align_and_load(sensor_name, limit_run=False) # Full run by default now
    
    if len(X) == 0:
        print(f"No data found for {sensor_name}. Skipping.")
        return
        
    print(f"Training on {len(X)} samples for {sensor_name}...")
    
    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train XGBoost
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    
    # 4. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy for {sensor_name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # 5. Save
    os.makedirs("models", exist_ok=True)
    model_path = f"models/behavior_{sensor_name}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # 6. Report Summary
    with open(f"models/report_{sensor_name}.txt", "w") as f:
        f.write(f"Sensor: {sensor_name}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))

def main():
    sensors_to_train = ["weather"]
    
    for sensor in sensors_to_train:
        try:
            train_sensor_model(sensor)
        except Exception as e:
            print(f"Error training {sensor}: {e}")

if __name__ == "__main__":
    main()
