from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class IMUData(BaseModel):
    accel: List[float] = Field(..., description="[x, y, z] acceleration")
    gyro: List[float] = Field(..., description="[x, y, z] gyroscope")

class SensorPayloadV1(BaseModel):
    device_id: str
    schema_version: str = "1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    imu: IMUData
    temperature: float
    rumination_score: Optional[int] = 0

    class Config:
        json_schema_extra = {
            "example": {
                "device_id": "COW-001",
                "schema_version": "1.0",
                "timestamp": "2023-10-27T10:00:00Z",
                "imu": {
                    "accel": [0.1, 0.2, 9.8],
                    "gyro": [0.01, 0.02, 0.03]
                },
                "temperature": 38.5,
                "rumination_score": 45
            }
        }
