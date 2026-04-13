from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.schemas.v1 import SensorPayloadV1
from app.models.db_models import SensorData, Alert
from app.services.preprocessing import Preprocessor
from app.services.features import FeatureExtractor
from app.services.anomaly import AnomalyDetector
from app.services.rule_engine import RuleEngine

router = APIRouter()
detector = AnomalyDetector()

@router.post("/ingest")
async def ingest_sensor_data(payload: SensorPayloadV1, db: Session = Depends(get_db)):
    # 1. Store the raw data
    new_data = SensorData(
        device_id=payload.device_id,
        schema_version=payload.schema_version,
        raw_payload=payload.model_dump(),
        temperature=payload.temperature,
        rumination_score=payload.rumination_score
    )
    db.add(new_data)
    db.commit()
    db.refresh(new_data)
    
    # 2. Pipeline: Preprocessing + Feature Extraction
    window_df = Preprocessor.get_window_from_db(db, payload.device_id)
    
    response = {
        "status": "success",
        "data_id": new_data.id,
        "alerts": []
    }
    
    if window_df is not None:
        features = FeatureExtractor.compute_features(window_df)
        
        # 3. Running AI Anomaly Detection
        anomaly_score = detector.predict(features)
        
        # 4. Running Rule Engine
        detected_alerts = RuleEngine.evaluate(new_data, features, anomaly_score)
        
        # 5. Store Alerts in DB and format response
        for alert_data in detected_alerts:
            alert_db = Alert(
                device_id=payload.device_id,
                alert_type=alert_data["type"],
                severity=alert_data["severity"],
                message=alert_data["message"]
            )
            db.add(alert_db)
            response["alerts"].append(alert_data)
        
        db.commit()
        response["anomaly_detected"] = (anomaly_score == -1)
    
    return response

@router.get("/alerts/{device_id}")
async def get_alerts(device_id: str, db: Session = Depends(get_db)):
    alerts = db.query(Alert).filter(Alert.device_id == device_id).order_by(Alert.timestamp.desc()).limit(10).all()
    return alerts
