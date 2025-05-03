"""
Notion Agent using LangChain and LangGraph

This agent provides tools to interact with Notion API using LangChain and LangGraph.
"""

from typing import Dict, List, Optional, Type
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

load_dotenv()

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
        response = requests.get(
            f"{get_base_url()}/pages/{tool_input}",
            headers=get_notion_headers()
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to fetch page {tool_input}: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)

@tool("get_notion_page_blocks")
def get_page_blocks_tool(tool_input: str) -> List[Dict]:
    """Fetch all blocks from a Notion page.
    
    Args:
        tool_input (str): The ID of the Notion page to fetch blocks from
        
    Returns:
        List[Dict]: List of blocks from the Notion page
    """
    try:
        blocks = []
        has_more = True
        start_cursor = None

        while has_more:
            params = {}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = requests.get(
                f"{get_base_url()}/blocks/{tool_input}/children",
                headers=get_notion_headers(),
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            blocks.extend(data["results"])
            has_more = data["has_more"]
            start_cursor = data.get("next_cursor")

        return blocks
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to fetch blocks for page {tool_input}: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)

@tool("get_notion_page_comments")
def get_page_comments_tool(tool_input: str) -> List[Dict]:
    """Fetch all comments from a Notion page.
    
    Args:
        tool_input (str): The ID of the Notion page to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion page
    """
    try:
        comments = []
        has_more = True
        start_cursor = None

        while has_more:
            params = {"block_id": tool_input}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = requests.get(
                f"{get_base_url()}/comments",
                headers=get_notion_headers(),
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            comments.extend(data["results"])
            has_more = data["has_more"]
            start_cursor = data.get("next_cursor")

        return comments
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to fetch comments for page {tool_input}: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)

@tool("get_notion_block_comments")
def get_block_comments_tool(tool_input: str) -> List[Dict]:
    """Fetch all comments from a specific Notion block.
    
    Args:
        tool_input (str): The ID of the Notion block to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion block
    """
    try:
        comments = []
        has_more = True
        start_cursor = None

        while has_more:
            params = {"block_id": tool_input}
            if start_cursor:
                params["start_cursor"] = start_cursor

            response = requests.get(
                f"{get_base_url()}/comments",
                headers=get_notion_headers(),
                params=params
            )
            response.raise_for_status()
            data = response.json()
            
            # Filter comments that belong to the specific block
            block_comments = [comment for comment in data["results"] 
                            if comment.get("parent", {}).get("block_id") == tool_input]
            comments.extend(block_comments)
            
            has_more = data["has_more"]
            start_cursor = data.get("next_cursor")

        return comments
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to fetch comments for block {tool_input}: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)

@tool("insert_notion_comment")
def insert_comment_tool(tool_input: NotionCommentInput) -> Dict:
    """Insert a comment into a Notion block.
    
    Args:
        tool_input (NotionCommentInput): Input containing block_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    try:
        block_id = tool_input.block_id
        text = tool_input.text
            
        payload = {
            "parent": {
                "block_id": block_id
            },
            "rich_text": [
                {
                    "text": {
                        "content": text
                    }
                }
            ]
        }
        
        response = requests.post(
            f"{get_base_url()}/comments",
            headers=get_notion_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to insert comment: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)
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
        response = requests.get(
            f"{get_base_url()}/pages/{tool_input}",
            headers=get_notion_headers()
        )
        response.raise_for_status()
        page_data = response.json()
        
        # Extract title from page properties
        title_property = page_data.get("properties", {}).get("title", {})
        if not title_property:
            raise Exception("Title property not found in page")
            
        title_text = title_property.get("title", [])
        if not title_text:
            return ""
            
        return title_text[0].get("plain_text", "")
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to fetch page title for {tool_input}: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)

@tool("insert_notion_page_comment")
def insert_page_comment_tool(tool_input: NotionPageCommentInput) -> Dict:
    """Insert a comment into a Notion page.
    
    Args:
        tool_input (NotionPageCommentInput): Input containing page_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    try:
        page_id = tool_input.page_id
        text = tool_input.text
            
        payload = {
            "parent": {
                "page_id": page_id
            },
            "rich_text": [
                {
                    "text": {
                        "content": text
                    }
                }
            ]
        }
        
        response = requests.post(
            f"{get_base_url()}/comments",
            headers=get_notion_headers(),
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_detail = f"Failed to insert page comment: {str(e)}"
        if hasattr(e.response, 'json'):
            error_detail += f" - {e.response.json()}"
        raise Exception(error_detail)
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
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

def run_notion_agent(query: str, chat_history: Optional[List] = None) -> str:
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
        "chat_history": chat_history
    })
    
    return result["output"]

# Example usage
if __name__ == "__main__":
    # Example query
    query = "Add comment to the page with id 1e7ff0df284780d0973bf7d70305a2f4 with text 'This is a test comment'"
    response = run_notion_agent(query)
    print(response)
