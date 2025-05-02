from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

app = FastAPI(
    title="Korean Teacher Agent",
    description="A dummy FastAPI application for Korean Teacher Agent",
    version="1.0.0"
)

class HealthResponse(BaseModel):
    status: str
    version: str

@app.get("/")
async def root():
    return {"message": "Welcome to Korean Teacher Agent API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/dummy-data")
async def get_dummy_data():
    return {
        "korean_phrases": [
            "안녕하세요 (Hello)",
            "감사합니다 (Thank you)",
            "미안합니다 (I'm sorry)"
        ],
        "count": 3
    }
