from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
import logging
import json

from src.agents.notion_agent import notion_client

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


class NotionWebhookData(BaseModel):
    updated_blocks: list = []
    parent: dict = {}

class NotionEntity(BaseModel):
    id: str
    type: str

class NotionWebhookRequest(BaseModel):
    id: str
    timestamp: str
    workspace_id: str
    workspace_name: str
    subscription_id: str
    integration_id: str
    type: str
    authors: list
    accessible_by: list = []  # Make this field optional with default empty list
    attempt_number: int
    entity: NotionEntity
    data: NotionWebhookData


@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        body = await request.json()
        webhook_data = NotionWebhookRequest(**body)
        logger.info(f"Received Notion webhook: {webhook_data.type}")
        logger.info(f"Entity ID: {webhook_data.entity.id}, Type: {webhook_data.entity.type}")
        logger.info(f"Webhook data: {webhook_data.model_dump_json()}")
        
        if webhook_data.type == "page.properties_updated":
            page_id = webhook_data.entity.id
            page_data = notion_client.get_page(page_id)
            logger.info(f"Page data: {page_data}")
            page_properties = page_data['properties']
            logger.info(f"Page properties: {page_properties}")
            if "Status" in page_properties:
                status_property = page_properties["Status"]
                logger.info(f"Status property: {status_property}")
                if status_property['status']['name'] == "Review requested":
                    logger.info(f"Page property is in review")
                    logger.info(f"Page property: {status_property}")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {"status": "error", "message": str(e)}


