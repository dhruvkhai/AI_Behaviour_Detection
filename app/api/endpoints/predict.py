from fastapi import APIRouter, HTTPException
import numpy as np
import pandas as pd
import json
import os
from scipy.stats import skew, kurtosis
from app.models.schemas.predict_schema import PredictPayload
from app.ml.classifiers import MasterFusionClassifier, DeepBehaviorClassifier
from app.services.anomaly import AnomalyDetector

router = APIRouter()

# Initialize models globally so they're in-memory across requests
MASTER_MODEL = MasterFusionClassifier()
# Ensure Deep Model doesn't crash if CUDA isn't set up on Render tier
try:
    DEEP_MODEL = DeepBehaviorClassifier()
except Exception as e:
    print(f"[Warning] Deep Model failed to load: {e}")
    DEEP_MODEL = None

ANOMALY_DETECTOR = AnomalyDetector()

# Load encoder to return human-readable labels and feature schemas
ENCODER_PATH = "models/behavior_master_label_encoder.json"
FEATURES_PATH = "models/behavior_master_features.json"

if os.path.exists(ENCODER_PATH):
    with open(ENCODER_PATH, "r") as f:
        LABEL_MAP = json.load(f)
        # Reverse map {string: intId} back to {intId: string}
        INV_LABEL_MAP = {int(v): k for k, v in LABEL_MAP.items()}
else:
    INV_LABEL_MAP = {0: "Grazing", 1: "Ruminating", 2: "Resting", 3: "Walking", 4: "Other"}

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, "r") as f:
        FEATURE_NAMES = json.load(f)
else:
    # Dummy generic feature names if the file is missing from a fresh clone
    FEATURE_NAMES = ["immu_accel_x_mps2_mean", "immu_accel_y_mps2_mean", "cbt_temperature_C_mean"]

def robust_stats(arr):
    """Calculates [mean, std, min, max, skew, kurt] handling empty or small lists."""
    if not arr or len(arr) == 0:
        return [np.nan] * 6
    if len(arr) == 1:
        val = arr[0]
        return [val, 0.0, val, val, 0.0, 0.0]
    
    a = np.array(arr)
    return [
        np.mean(a),
        np.std(a),
        np.min(a),
        np.max(a),
        skew(a) if len(a) > 2 else 0.0,
        kurtosis(a) if len(a) > 2 else 0.0
    ]

def extract_features(payload: PredictPayload):
    """
    Transforms the JSON payload into an exact 1xN feature vector matching XGBoost schema.
    Also handles missing variables gracefully (NaNs).
    """
    # 1. Parse arrays
    motion_x = [m[0] for m in payload.motion] if len(payload.motion) > 0 and len(payload.motion[0]) > 0 else []
    motion_y = [m[1] for m in payload.motion] if len(payload.motion) > 0 and len(payload.motion[0]) > 1 else []
    motion_z = [m[2] for m in payload.motion] if len(payload.motion) > 0 and len(payload.motion[0]) > 2 else []
    temp_array = [payload.temperature] if payload.temperature else []
    
    # 2. Extract stats mapping to prefix
    stats_dict = {}
    
    # Motion (Accel X, Y, Z) - standardizing mapping to what was likely used in Master Fusion (immu_accel)
    m_x_stats = robust_stats(motion_x)
    m_y_stats = robust_stats(motion_y)
    m_z_stats = robust_stats(motion_z)
    temp_stats = robust_stats(temp_array)
    
    suffixes = ["_mean", "_std", "_min", "_max", "_skew", "_kurt"]
    
    for i, s in enumerate(suffixes):
        stats_dict[f"immu_accel_x_mps2{s}"] = m_x_stats[i]
        stats_dict[f"immu_accel_y_mps2{s}"] = m_y_stats[i]
        stats_dict[f"immu_accel_z_mps2{s}"] = m_z_stats[i]
        stats_dict[f"cbt_temperature_C{s}"] = temp_stats[i]
        # Any missing sensor like UWB, Milk, THI will naturally not be populated here
        
    # 3. Build fully aligned 1xN feature vector against the Master Model schema
    final_vector = []
    for feat in FEATURE_NAMES:
        final_vector.append(stats_dict.get(feat, np.nan))
        
    return np.array(final_vector), stats_dict


@router.post("/predict")
async def predict_behavior(payload: PredictPayload):
    """
    Unified Inference Endpoint.
    Consumes raw JSON, computes dynamic features, degrades gracefully over missing inputs,
    and queries all active ML models (XGBoost + Anomaly + Deep Learning).
    """
    try:
        # 1. Preprocessing Pipeline
        features_np, _ = extract_features(payload)
        features_df = pd.DataFrame([features_np], columns=FEATURE_NAMES)
        
        # 2. XGBoost Classification (Master Fusion)
        master_pred_id = -1
        master_conf = 0.0
        if MASTER_MODEL.model is not None:
            preds = MASTER_MODEL.predict(features_df)
            probs = MASTER_MODEL.predict_proba(features_df)
            master_pred_id = int(preds[0])
            master_conf = float(np.max(probs[0]))
        
        # 3. Anomaly Detection (Isolation Forest)
        # We fill NaNs with 0 for isolation forest since it doesn't like NaNs inherently like XGBoost does
        features_clean = features_df.fillna(0.0)
        anomaly_score = ANOMALY_DETECTOR.predict(features_clean)
        is_anomaly = anomaly_score == -1
        
        # 4. (Optional) Deep Learning PyTorch sequence inference
        # If the Deep Model is loaded, we synthesize a sequence input (channels = len(FEATURE_NAMES))
        # Since PyTorch expects a sequence length >= 3 for convolution kernels, we stack our extracted features
        dl_pred_id = None
        if DEEP_MODEL is not None and DEEP_MODEL.model is not None:
            # Reshape specifically for DeepBehaviorClassifier: (channels, seq_len)
            # Impute NaN out for neural net
            clean_tensor_np = np.nan_to_num(features_np, nan=0.0)
            clean_tensor_np = np.tile(clean_tensor_np, (3, 1)).T  # shape (channels, 3)
            
            try:
                dl_preds = DEEP_MODEL.predict(clean_tensor_np)
                dl_pred_id = int(dl_preds[0])
            except Exception as e:
                print(f"[Warning] PyTorch Inference error: {e}")
        
        # 5. Format Response
        predicted_class_id = master_pred_id if master_pred_id != -1 else (dl_pred_id if dl_pred_id is not None else 0)
        predicted_label = INV_LABEL_MAP.get(predicted_class_id, "Unknown")
        
        return {
            "cow_id": payload.cow_id,
            "status": "success",
            "prediction": {
                "behaviour": predicted_label,
                "confidence": round(master_conf, 4),
                "anomaly": is_anomaly
            },
            "system": {
                "models_executed": {
                    "master_xgboost": MASTER_MODEL.model is not None,
                    "deep_learning_cnn": dl_pred_id is not None,
                    "isolation_forest": True
                }
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
