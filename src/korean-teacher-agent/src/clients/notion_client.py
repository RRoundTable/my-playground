"""
NotionAPIClient for interacting with the Notion API.

This module provides a client for interacting with the Notion API.
"""

from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
import requests

load_dotenv()

class NotionAPIError(Exception):
    """Custom exception for Notion API errors."""
    def __init__(self, message: str, response: Optional[requests.Response] = None):
        super().__init__(message)
        self.response = response

class NotionAPIClient:
    """Client for interacting with the Notion API."""
    
    def __init__(self, token: Optional[str] = None):
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {token or os.getenv('NOTION_TOKEN')}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
    
    def _handle_response(self, response: requests.Response) -> Any:
        """Handle API response and raise appropriate exceptions."""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_detail = str(e)
            if hasattr(e.response, 'json'):
                error_detail += f" - {e.response.json()}"
            raise NotionAPIError(error_detail, e.response)
    
    def _paginated_request(self, url: str, params: Optional[Dict] = None) -> List[Dict]:
        """Make a paginated request to the Notion API."""
        results = []
        has_more = True
        start_cursor = None
        
        while has_more:
            if start_cursor:
                params = params or {}
                params["start_cursor"] = start_cursor
            
            response = requests.get(url, headers=self.headers, params=params)
            data = self._handle_response(response)
            
            results.extend(data["results"])
            has_more = data["has_more"]
            start_cursor = data.get("next_cursor")
        
        return results
    
    def get_page(self, page_id: str) -> Dict:
        """Get a Notion page by ID."""
        response = requests.get(
            f"{self.base_url}/pages/{page_id}",
            headers=self.headers
        )
        return self._handle_response(response)
    
    def get_page_blocks(self, page_id: str) -> List[Dict]:
        """Get all blocks from a Notion page."""
        return self._paginated_request(
            f"{self.base_url}/blocks/{page_id}/children"
        )
    
    def get_comments(self, block_id: str = None) -> List[Dict]:
        """Get comments from a block or page.
        
        For page comments, we first get the page's blocks and then fetch comments for each block.
        """
        return self._paginated_request(
            f"{self.base_url}/comments",
            params={"block_id": block_id}
        )
    
    def insert_comment(self, text: str, block_id: Optional[str] = None, page_id: Optional[str] = None) -> Dict:
        """Insert a comment into a block or page."""
        if not block_id and not page_id:
            raise ValueError("Either block_id or page_id must be provided")
            
        parent = {"block_id": block_id} if block_id else {"page_id": page_id}
        payload = {
            "parent": parent,
            "rich_text": [{"text": {"content": text}}]
        }
        
        response = requests.post(
            f"{self.base_url}/comments",
            headers=self.headers,
            json=payload
        )
        return self._handle_response(response)
    
    def update_page_properties(self, page_id: str, properties: Dict) -> Dict:
        """Update properties of a Notion page.
        
        Args:
            page_id (str): The ID of the Notion page
            properties (Dict): Dictionary of properties to update
            
        Returns:
            Dict: The updated page data from Notion API
        """
        payload = {
            "properties": properties
        }
        
        response = requests.patch(
            f"{self.base_url}/pages/{page_id}",
            headers=self.headers,
            json=payload
        )
        return self._handle_response(response) 