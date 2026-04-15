from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "Cow Monitoring System"
    DATABASE_URL: str = "sqlite:///./data/cow_monitor.db"  
    API_V1_STR: str = "/api/v1"
    
    # AI Pipeline Config
    WINDOW_SIZE: int = 30  # Number of samples per window
    ANOMALY_THRESHOLD: float = -0.5  # Isolation Forest decision threshold
    
    class Config:
        env_file = ".env"

settings = Settings()
