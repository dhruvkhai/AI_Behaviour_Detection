from fastapi import FastAPI
from app.api.endpoints import ingest
from app.core.config import settings
from app.core.database import engine, Base

# Initialize Database
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Cow Behaviour Detection System Prototype"
)

# Include Routers
app.include_router(ingest.router, prefix=settings.API_V1_STR, tags=["Ingestion"])

@app.get("/")
async def health_check():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
