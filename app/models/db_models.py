from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.sql import func
from app.core.database import Base

class SensorData(Base):
    __tablename__ = "sensor_data"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    schema_version = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Store raw readings as JSON for flexibility (can be indexed if moved to Postgres later)
    # This allows changing sensor fields without breaking the schema
    raw_payload = Column(JSON)
    
    # Pre-extracted features for quick analysis if needed
    temperature = Column(Float)
    rumination_score = Column(Float)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    alert_type = Column(String)  # 'HIGH_TEMP', 'LOW_ACTIVITY', 'ANOMALY'
    severity = Column(String)    # 'LOW', 'MEDIUM', 'HIGH'
    message = Column(String)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_resolved = Column(Integer, default=0)
