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
    
    try:
        # 1. 페이지 정보 가져오기
        print("\n1. 페이지 정보 가져오기:")
        page_info = client.get_page(test_page_id)
        print(f"페이지 제목: {page_info.get('properties', {}).get('title', {}).get('title', [{}])[0].get('plain_text', 'Unknown')}")
        print(f"페이지 ID: {page_info.get('id')}")
        print(f"생성 시간: {page_info.get('created_time')}")
        print(f"최종 수정 시간: {page_info.get('last_edited_time')}")
        
        # 2. 페이지 블록 가져오기
        print("\n2. 페이지 블록 목록 가져오기:")
        blocks = client.get_page_blocks(test_page_id)
        print(f"총 {len(blocks)}개의 블록을 찾았습니다.")
        
        if blocks:
            print("\n첫 번째 블록 정보:")
            first_block = blocks[0]
            block_id = first_block.get("id")
            block_type = first_block.get("type")
            print(f"블록 ID: {block_id}")
            print(f"블록 타입: {block_type}")
            print(f"블록 내용: {json.dumps(first_block.get(block_type, {}), indent=2, ensure_ascii=False)}")
            
            # 3. 블록에 코멘트 추가하기
            print("\n3. 블록에 코멘트 추가하기:")
            comment_text = "이것은 API 테스트를 위한 코멘트입니다."
            comment = client.insert_comment(comment_text, block_id=block_id)
            print(f"코멘트 ID: {comment.get('id')}")
            print(f"코멘트 내용: {comment.get('rich_text', [{}])[0].get('text', {}).get('content')}")
            print(f"생성 시간: {comment.get('created_time')}")
            
            # 4. 블록의 코멘트 가져오기
            print("\n4. 블록의 코멘트 가져오기:")
            comments = client.get_comments(block_id=block_id)
            print(f"총 {len(comments)}개의 코멘트를 찾았습니다.")
            if comments:
                print("\n최근 코멘트 내용:")
                for i, comment in enumerate(comments[:3], 1):
                    comment_content = comment.get('rich_text', [{}])[0].get('text', {}).get('content', 'No content')
                    print(f"코멘트 {i}: {comment_content}")
        
        # 5. 페이지에 코멘트 추가하기
        print("\n5. 페이지에 코멘트 추가하기:")
        page_comment_text = "이것은 페이지 전체에 대한 API 테스트 코멘트입니다."
        page_comment = client.insert_comment(page_comment_text, page_id=test_page_id)
        print(f"페이지 코멘트 ID: {page_comment.get('id')}")
        print(f"페이지 코멘트 내용: {page_comment.get('rich_text', [{}])[0].get('text', {}).get('content')}")
        
        # 6. 페이지 속성 업데이트하기
        print("\n6. 페이지 속성 업데이트하기:")
        # 참고: properties 구조는 페이지의 데이터베이스 구조에 따라 달라집니다.
        # 아래는 일반적인 Title 속성을 가진 페이지의 예시입니다.
        properties = {
            "Title": {
                "title": [
                    {
                        "type": "text",
                        "text": {
                            "content": "API로 업데이트된 페이지 제목"
                        }
                    }
                ]
            }
        }
        updated_page = client.update_page_properties(test_page_id, properties)
        print("페이지 속성이 업데이트되었습니다.")
        print(f"업데이트된 제목: {updated_page.get('properties', {}).get('Title', {}).get('title', [{}])[0].get('plain_text', 'Unknown')}")
        print(f"최종 수정 시간: {updated_page.get('last_edited_time')}")
        
    except NotionAPIError as e:
        print(f"Notion API 에러 발생: {e}")
        if e.response:
            print(f"상태 코드: {e.response.status_code}")
            try:
                print(f"에러 상세: {e.response.json()}")
            except:
                print(f"응답 본문: {e.response.text}")
    except Exception as e:
        print(f"일반 에러 발생: {e}")
    
    print("\n" + "=" * 50)
    print("NotionAPIClient 테스트 완료")
    print("=" * 50)

    print("\n테스트 실행 방법:")
    print("1. '.env' 파일에 다음 환경 변수를 설정하세요:")
    print("   NOTION_TOKEN=your_notion_integration_token")
    print("   TEST_PAGE_ID=your_test_page_id")
    print("2. 다음 명령어로 테스트를 실행하세요:")
    print("   python notion_client.py")

