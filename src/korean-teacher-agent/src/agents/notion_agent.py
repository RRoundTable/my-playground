"""
Notion Agent

Notion agent is a tool that allows you to create, read, update, and delete Notion pages.
"""

from typing import Dict, List, Optional
import requests
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

class NotionAgent:
    def __init__(self):
        self.notion_token = os.getenv("NOTION_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.notion_token}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        self.base_url = "https://api.notion.com/v1"
        self.app = FastAPI(
            title="Notion Agent API",
            description="API for handling Notion webhooks and page operations",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            servers=[
                {"url": "https://ai-agent.nocoders.ai", "description": "Production server"},
                {"url": "http://localhost:8000", "description": "Local development server"}
            ]
        )
        
        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # In production, replace with specific origins
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.setup_webhook()

    def get_page(self, page_id: str) -> Dict:
        """Fetch a Notion page by its ID."""
        try:
            response = requests.get(
                f"{self.base_url}/pages/{page_id}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=str(e))

    def get_page_blocks(self, page_id: str) -> List[Dict]:
        """Fetch all blocks from a Notion page."""
        try:
            blocks = []
            has_more = True
            start_cursor = None

            while has_more:
                params = {}
                if start_cursor:
                    params["start_cursor"] = start_cursor

                response = requests.get(
                    f"{self.base_url}/blocks/{page_id}/children",
                    headers=self.headers,
                    params=params
                )
                response.raise_for_status()
                data = response.json()
                
                blocks.extend(data["results"])
                has_more = data["has_more"]
                start_cursor = data.get("next_cursor")

            return blocks
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=str(e))

    def setup_webhook(self):
        """Setup webhook endpoint for Notion integration."""
        @self.app.post("/notion-webhook")
        async def handle_notion_webhook(request: Request):
            try:
                # Get the webhook payload
                payload = await request.json()
                print(f"Received webhook payload: {payload}")

                # Extract page ID from the payload
                # Notion webhook payload structure:
                # {
                #   "object": "page",
                #   "id": "page_id",
                #   "created_time": "...",
                #   "last_edited_time": "...",
                #   "parent": {...},
                #   "archived": false,
                #   "properties": {...}
                # }
                page_id = payload.get("id")
                if not page_id:
                    return {"status": "error", "message": "No page ID in payload"}

                # Get the page details
                page = self.get_page(page_id)
                print(f"Retrieved page: {page}")

                # Get all blocks from the page
                blocks = self.get_page_blocks(page_id)
                print(f"Retrieved {len(blocks)} blocks")

                # Check if the page has a review tag
                if self._has_review_tag(page):
                    print("Page has review tag, processing...")
                    # Here you can add your custom logic for handling review pages
                    return {
                        "status": "success",
                        "message": "Review page processed",
                        "page": page,
                        "blocks": blocks
                    }
                
                return {
                    "status": "success",
                    "message": "Page processed",
                    "page": page,
                    "blocks": blocks
                }
            
            except Exception as e:
                print(f"Error processing webhook: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))

    def _has_review_tag(self, page: Dict) -> bool:
        """Check if the page has a review tag."""
        try:
            properties = page.get("properties", {})
            
            # Check for tags in different possible property names
            for prop_name in ["Tags", "tags", "Tag", "tag"]:
                tags_property = properties.get(prop_name)
                if tags_property and tags_property.get("type") == "multi_select":
                    tags = tags_property.get("multi_select", [])
                    if any(tag.get("name", "").lower() == "review" for tag in tags):
                        return True
            
            return False
            
        except Exception as e:
            print(f"Error checking review tag: {str(e)}")
            return False

    def run_webhook_server(self, host: str = "0.0.0.0", port: int = 8000, ssl_keyfile: str = None, ssl_certfile: str = None):
        """Run the FastAPI webhook server with optional SSL configuration."""
        import uvicorn
        
        ssl_config = {}
        if ssl_keyfile and ssl_certfile:
            ssl_config = {
                "ssl_keyfile": ssl_keyfile,
                "ssl_certfile": ssl_certfile
            }
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            **ssl_config
        )

# Test section
if __name__ == "__main__":
    # Initialize NotionAgent
    notion_agent = NotionAgent()
    
    # Test webhook server
    print("Starting webhook server...")
    print("Server will run on https://ai-agent.nocoders.ai")
    print("Press Ctrl+C to stop the server")
    
    try:
        # For production, you would use SSL certificates
        # notion_agent.run_webhook_server(
        #     ssl_keyfile="path/to/privkey.pem",
        #     ssl_certfile="path/to/cert.pem"
        # )
        
        # For local development
        notion_agent.run_webhook_server()
    except KeyboardInterrupt:
        print("\nServer stopped")
