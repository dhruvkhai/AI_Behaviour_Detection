from pydantic import BaseModel, Field
from typing import List, Optional

class PredictPayload(BaseModel):
    cow_id: str = Field(..., example="RFID_123")
    
    # We accept arrays mapping to a window to keep the API stateless and fast.
    motion: List[List[float]] = Field(
        default_factory=list,
        example=[[0.1, 0.2, 9.8, 0.01, 0.02, 0.03], [0.1, 0.1, 9.7, 0.0, 0.01, 0.02]],
        description="List of raw [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] readings forming a time window."
    )
    
    rumination: List[float] = Field(
        default_factory=list,
        example=[0.5, 0.6],
        description="List of rumination scores corresponding to the motion readings."
    )
    
    temperature: Optional[float] = Field(38.5, example=38.5)
    timestamp: Optional[str] = Field(None, example="2023-10-27T10:00:00Z")

    class Config:
        json_schema_extra = {
            "example": {
                "cow_id": "RFID_123",
                "motion": [
                    [0.1, 0.2, 9.8],
                    [0.11, 0.21, 9.81]
                ],
                "rumination": [45, 46],
                "temperature": 38.5,
                "timestamp": "2023-10-27T10:00:00Z"
            }
        }
