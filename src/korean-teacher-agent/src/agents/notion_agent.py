"""
Notion Agent using LangChain and LangGraph

This agent provides tools to interact with Notion API using LangChain and LangGraph.
"""

from typing import Dict, List, Optional, Type, Any
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
import requests
from src.prompts import create_notion_agent_prompt
from langgraph.prebuilt import create_react_agent

load_dotenv()

class NotionAPIError(Exception):
    """Custom exception for Notion API errors."""
    def __init__(self, message: str, response: Optional[requests.Response] = None):
        super().__init__(message)
        self.response = response

class NotionAPIClient:
    """Client for interacting with the Notion API."""
    
    def __init__(self):
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('NOTION_TOKEN')}",
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

# Initialize the API client
notion_client = NotionAPIClient()

class NotionPageInput(BaseModel):
    """Input for Notion page operations."""
    page_id: str = Field(..., description="The ID of the Notion page")

class NotionCommentInput(BaseModel):
    """Input for Notion comment operations."""
    block_id: str = Field(..., description="The ID of the Notion block to comment on")
    text: str = Field(..., description="The text content of the comment")

class NotionPageCommentInput(BaseModel):
    """Input for Notion page comment operations."""
    page_id: str = Field(..., description="The ID of the Notion page to comment on")
    text: str = Field(..., description="The text content of the comment")

def get_notion_headers():
    """Get Notion API headers."""
    notion_token = os.getenv("NOTION_TOKEN")
    return {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

def get_base_url():
    """Get Notion API base URL."""
    return "https://api.notion.com/v1"

@tool("get_notion_page")
def get_page_tool(tool_input: str) -> Dict:
    """Fetch a Notion page by its ID.
    
    Args:
        tool_input (str): The ID of the Notion page to fetch
        
    Returns:
        Dict: The page data from Notion API
    """
    try:
        return notion_client.get_page(tool_input)
    except NotionAPIError as e:
        raise Exception(f"Failed to fetch page {tool_input}: {str(e)}")

@tool("get_notion_page_blocks")
def get_page_blocks_tool(tool_input: str) -> List[Dict]:
    """Fetch all blocks from a Notion page.
    
    Args:
        tool_input (str): The ID of the Notion page to fetch blocks from
        
    Returns:
        List[Dict]: List of blocks from the Notion page
    """
    try:
        return notion_client.get_page_blocks(tool_input)
    except NotionAPIError as e:
        raise Exception(f"Failed to fetch blocks for page {tool_input}: {str(e)}")

@tool("get_notion_page_comments")
def get_page_comments_tool(tool_input: str) -> List[Dict]:
    """Fetch all comments from a Notion page.
    
    Args:
        tool_input (str): The ID of the Notion page to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion page
    """
    try:
        return notion_client.get_comments(block_id=tool_input)
    except NotionAPIError as e:
        raise Exception(f"Failed to fetch comments for page {tool_input}: {str(e)}")

@tool("get_notion_block_comments")
def get_block_comments_tool(tool_input: str) -> List[Dict]:
    """Fetch all comments from a specific Notion block.
    
    Args:
        tool_input (str): The ID of the Notion block to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion block
    """
    try:
        return notion_client.get_comments(block_id=tool_input)
    except NotionAPIError as e:
        raise Exception(f"Failed to fetch comments for block {tool_input}: {str(e)}")

@tool("insert_notion_comment")
def insert_comment_tool(tool_input: NotionCommentInput) -> Dict:
    """Insert a comment into a Notion block.
    
    Args:
        tool_input (NotionCommentInput): Input containing block_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    try:
        return notion_client.insert_comment(
            text=tool_input.text,
            block_id=tool_input.block_id
        )
    except NotionAPIError as e:
        raise Exception(f"Failed to insert comment: {str(e)}")
    except ValueError as e:
        raise Exception(f"Invalid input: {str(e)}")

@tool("get_notion_page_title")
def get_page_title_tool(tool_input: str) -> str:
    """Get the title of a Notion page.
    
    Args:
        tool_input (str): The ID of the Notion page
        
    Returns:
        str: The title of the Notion page
    """
    try:
        page_data = notion_client.get_page(tool_input)
        title_property = page_data.get("properties", {}).get("title", {})
        if not title_property:
            raise Exception("Title property not found in page")
            
        title_text = title_property.get("title", [])
        if not title_text:
            return ""
            
        return title_text[0].get("plain_text", "")
    except NotionAPIError as e:
        raise Exception(f"Failed to fetch page title for {tool_input}: {str(e)}")

@tool("insert_notion_page_comment")
def insert_page_comment_tool(tool_input: NotionPageCommentInput) -> Dict:
    """Insert a comment into a Notion page.
    
    Args:
        tool_input (NotionPageCommentInput): Input containing page_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    try:
        return notion_client.insert_comment(
            text=tool_input.text,
            page_id=tool_input.page_id
        )
    except NotionAPIError as e:
        raise Exception(f"Failed to insert page comment: {str(e)}")
    except ValueError as e:
        raise Exception(f"Invalid input: {str(e)}")

def create_notion_agent():
    """Create and return a LangChain agent with Notion tools."""
    llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
    
    tools = [
        get_page_tool,
        get_page_blocks_tool,
        get_page_comments_tool,
        get_block_comments_tool,
        insert_comment_tool,
        get_page_title_tool,
        insert_page_comment_tool
    ]
    
    prompt = create_notion_agent_prompt()

    return create_react_agent(model=llm, tools=tools, prompt=prompt, name="notion_agent")

def run_notion_agent(query: str, chat_history: Optional[List] = None, agent_scratchpad: Optional[List] = None) -> str:
    """Run the Notion agent with the given query and chat history.
    
    Args:
        query (str): The user's query
        chat_history (Optional[List]): Previous conversation history
        
    Returns:
        str: The agent's response
    """
    if chat_history is None:
        chat_history = []
        
    agent = create_notion_agent()
    result = agent.invoke({
        "input": query,
        "chat_history": chat_history,
        "agent_scratchpad": agent_scratchpad
    })
    
    return result["output"]
# Create the Notion agent instance
notion_agent = create_notion_agent()


# Example usage
if __name__ == "__main__":
    # Notion Agent 테스트 스크립트
    print("Notion Agent 테스트 시작")
    print("=" * 50)
    
    # 다양한 테스트 쿼리 준비
    test_queries = [
        "Get the title of page with id 1e7ff0df284780d0973bf7d70305a2f4",
        "Get all blocks from the page with id 1e7ff0df284780d0973bf7d70305a2f4",
    ]
    
    # 각 쿼리 실행 및 결과 출력
    for i, query in enumerate(test_queries, 1):
        print(f"\n테스트 #{i}: {query}")
        print("-" * 50)
        try:
            response = run_notion_agent(query)
            print(f"응답: {response}")
        except Exception as e:
            print(f"오류 발생: {str(e)}")
        print("-" * 50)
    
    print("\nNotion Agent 테스트 완료")
    print("=" * 50)
