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
    
    def get_paragraph_text_blocks(self, page_id: str) -> List[Dict[str, str]]:
        """Get only paragraph blocks with their plaintext content and ID.
        
        Args:
            page_id (str): The ID of the Notion page
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing 'id' and 'content' keys
        """
        blocks = self.get_page_blocks(page_id)
        paragraph_blocks = []
        
        for block in blocks:
            if block.get('type') == 'paragraph':
                content = ""
                rich_text = block.get('paragraph', {}).get('rich_text', [])
                
                for text_item in rich_text:
                    content += text_item.get('plain_text', '')
                
                if content:  # Only include if there's actual content
                    paragraph_blocks.append({
                        'id': block.get('id'),
                        'content': content
                    })
                    
        return paragraph_blocks
    
    def get_comment_content_blocks(self, block_id: Optional[str] = None, page_id: Optional[str] = None) -> List[Dict[str, str]]:
        """Get comments with their plaintext content and ID, plus parent information.
        
        Args:
            block_id (Optional[str]): The ID of the block to get comments from
            page_id (Optional[str]): The ID of the page to get comments from
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing 'id', 'content', and 'parent' keys
        """
        # Get comments from the block or page
        comments = self.get_comments(block_id)
        comment_blocks = []
        
        for comment in comments:
            content = ""
            rich_text = comment.get('rich_text', [])
            
            for text_item in rich_text:
                content += text_item.get('plain_text', '')
            
            parent_info = comment.get('parent', {})
            parent_type = parent_info.get('type')
            parent_id = None
            
            if parent_type == 'database_id':
                parent_id = parent_info.get('database_id')
            elif parent_type == 'page_id':
                parent_id = parent_info.get('page_id')
            elif parent_type == 'block_id':
                parent_id = parent_info.get('block_id')
            
            if content:  # Only include if there's actual content
                comment_blocks.append({
                    'id': comment.get('id'),
                    'content': content,
                    'parent_type': parent_type,
                    'parent_id': parent_id
                })
                
        return comment_blocks
    
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


if __name__ == "__main__":
    """Test script to demonstrate all NotionAPIClient functionality.
    
    환경 변수에 NOTION_TOKEN과 테스트용 PAGE_ID가 설정되어 있어야 합니다.
    """
    import json
    from pprint import pprint
    
    # Notion API 클라이언트 초기화
    client = NotionAPIClient()
    
    # 테스트용 페이지 ID (실제 존재하는 Notion 페이지 ID 필요)
    # .env 파일에 TEST_PAGE_ID 설정 또는 아래 직접 입력
    test_page_id = os.getenv("TEST_PAGE_ID", "1e9ff0df28478038a184fe3371797f96")
    
    if not test_page_id:
        print("ERROR: 테스트용 페이지 ID가 필요합니다. .env 파일에 TEST_PAGE_ID를 설정하세요.")
        exit(1)
    
    print("=" * 50)
    print("NotionAPIClient 테스트 시작")
    print("=" * 50)
    
    # 1. 페이지 정보 가져오기
    print("\n1. 페이지 정보 가져오기:")
    page_info = client.get_page(test_page_id)
    print(f"페이지 제목: {page_info.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Unknown')}")
    print(f"페이지 ID: {page_info.get('id')}")
    print(f"생성 시간: {page_info.get('created_time')}")
    print(f"최종 수정 시간: {page_info.get('last_edited_time')}")
    
    # 2. 페이지 블록 가져오기
    print("\n2. 페이지 block_id, content 가져오기:")
    blocks = client.get_paragraph_text_blocks(test_page_id)
    pprint(blocks)
    print(f"총 {len(blocks)}개의 블록을 찾았습니다.")

    # 3. 페이지 댓글 가져오기
    print("\n3. 페이지 댓글 가져오기:")
    comments = client.get_comment_content_blocks(test_page_id)
    pprint(comments)
