from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


class WebhookRequest(BaseModel):
    verification_token: str

@app.post("/webhook")
async def handle_webhook(request: WebhookRequest):
    logger.info(f"Received webhook with verification token: {request.verification_token}")
    logger.info(f"Webhook request body: {request.model_dump_json()}")
    return {"status": "success"}
