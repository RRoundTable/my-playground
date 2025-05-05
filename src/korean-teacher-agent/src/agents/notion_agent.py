"""
Notion Agent using LangChain and LangGraph

This agent provides tools to interact with Notion API using LangChain and LangGraph.
"""

from typing import Dict, List, Optional, Type, Any
import os
import logging
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
from src.clients.notion_client import NotionAPIClient, NotionAPIError
from src.agents.title_agent import evaluate_title_tool

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('notion_agent')

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

class NotionPagePropertyInput(BaseModel):
    """Input for Notion page property operations."""
    page_id: str = Field(..., description="The ID of the Notion page")
    property_id: str = Field(..., description="The ID of the property to get")

class NotionUpdatePagePropertiesInput(BaseModel):
    """Input for updating Notion page properties."""
    page_id: str = Field(..., description="The ID of the Notion page")
    properties: Dict = Field(..., description="Dictionary of properties to update")

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
def get_page_tool(page_id: str) -> Dict:
    """Fetch a Notion page by its ID.
    
    Args:
        page_id (str): The ID of the Notion page to fetch
        
    Returns:
        Dict: The page data from Notion API
    """
    logger.info(f"Tool used: get_notion_page with page_id: {page_id}")
    try:
        result = notion_client.get_page(page_id)
        logger.info(f"Tool output: get_notion_page returned data of length: {len(str(result))}")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch page {page_id}: {str(e)}")

@tool("get_notion_page_paragraph_text_blocks")
def get_page_paragraph_text_blocks_tool(page_id: str) -> List[Dict]:
    """Fetch all blocks from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch blocks from
        
    Returns:
        List[Dict]: List of blocks from the Notion page
    """
    logger.info(f"Tool used: get_notion_page_paragraph_text_blocks with page_id: {page_id}")
    try:
        result = notion_client.get_paragraph_text_blocks(page_id)
        logger.info(f"Tool output: get_notion_page_paragraph_text_blocks returned {len(result)} blocks")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch blocks for page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch blocks for page {page_id}: {str(e)}")

@tool("get_notion_page_comment_content_blocks")
def get_page_comment_content_blocks_tool(page_id: str) -> List[Dict]:
    """Fetch all comments from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion page
    """
    logger.info(f"Tool used: get_notion_page_comment_content_blocks with page_id: {page_id}")
    try:
        result = notion_client.get_comment_content_blocks(block_id=page_id)
        logger.info(f"Tool output: get_notion_page_comment_content_blocks returned {len(result)} comments")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch comments for page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch comments for page {page_id}: {str(e)}")

@tool("get_notion_block_comments")
def get_block_comments_tool(block_id: str) -> List[Dict]:
    """Fetch all comments from a specific Notion block.
    
    Args:
        block_id (str): The ID of the Notion block to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion block
    """
    logger.info(f"Tool used: get_notion_block_comments with block_id: {block_id}")
    try:
        result = notion_client.get_comments(block_id=block_id)
        logger.info(f"Tool output: get_notion_block_comments returned {len(result)} comments")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch comments for block {block_id}: {str(e)}")
        raise Exception(f"Failed to fetch comments for block {block_id}: {str(e)}")

@tool("insert_notion_comment")
def insert_comment_tool(tool_input: NotionCommentInput) -> Dict:
    """Insert a comment into a Notion block.
    
    Args:
        tool_input (NotionCommentInput): Input containing block_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    logger.info(f"Tool used: insert_notion_comment for block_id: {tool_input.block_id}")
    logger.info(f"Tool input: text length: {len(tool_input.text)}")
    try:
        result = notion_client.insert_comment(
            text=tool_input.text,
            block_id=tool_input.block_id
        )
        logger.info(f"Tool output: insert_notion_comment successfully created comment")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to insert comment: {str(e)}")
        raise Exception(f"Failed to insert comment: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise Exception(f"Invalid input: {str(e)}")

@tool("get_notion_page_title")
def get_page_title_tool(page_id: str) -> str:
    """Get the title of a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page
        
    Returns:
        str: The title of the Notion page
    """
    logger.info(f"Tool used: get_notion_page_title with page_id: {page_id}")
    try:
        page_data = notion_client.get_page(page_id)
        title_property = page_data.get("properties", {}).get("Title", {})
        if not title_property:
            logger.error(f"Title property not found in page {page_id}")
            raise Exception("Title property not found in page")
            
        title_text = title_property.get("title", [])
        if not title_text:
            logger.info("Tool output: get_notion_page_title returned empty title")
            return ""
        result = title_text[0].get("plain_text", "")
        logger.info(f"Tool output: get_notion_page_title returned title: '{result}'")
        return result
        
    except NotionAPIError as e:
        logger.error(f"Failed to fetch page title for {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch page title for {page_id}: {str(e)}")

@tool("insert_notion_page_comment")
def insert_page_comment_tool(tool_input: NotionPageCommentInput) -> Dict:
    """Insert a comment into a Notion page.
    
    Args:
        tool_input (NotionPageCommentInput): Input containing page_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    logger.info(f"Tool used: insert_notion_page_comment for page_id: {tool_input.page_id}")
    logger.info(f"Tool input: text length: {len(tool_input.text)}")
    try:
        result = notion_client.insert_comment(
            text=tool_input.text,
            page_id=tool_input.page_id
        )
        logger.info(f"Tool output: insert_notion_page_comment successfully created comment")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to insert page comment: {str(e)}")
        raise Exception(f"Failed to insert page comment: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise Exception(f"Invalid input: {str(e)}")


@tool("update_notion_page_properties")
def update_page_properties_tool(tool_input: NotionUpdatePagePropertiesInput) -> Dict:
    """Update properties of a Notion page.
    
    Args:
        tool_input (NotionUpdatePagePropertiesInput): Input containing page_id and properties
        
    Returns:
        Dict: The updated page data from Notion API
    """
    logger.info(f"Tool used: update_notion_page_properties for page_id: {tool_input.page_id}")
    logger.info(f"Tool input: properties keys: {list(tool_input.properties.keys())}")
    try:
        result = notion_client.update_page_properties(
            page_id=tool_input.page_id,
            properties=tool_input.properties
        )
        logger.info(f"Tool output: update_notion_page_properties successfully updated properties")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to update page properties: {str(e)}")
        raise Exception(f"Failed to update page properties: {str(e)}")


def create_notion_agent():
    """Create and return a LangChain agent with Notion tools."""
    logger.info("Creating notion agent")
    llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
    
    tools = [
        get_page_tool,
        get_page_paragraph_text_blocks_tool,
        get_page_comment_content_blocks_tool,
        get_block_comments_tool,
        insert_comment_tool,
        get_page_title_tool,
        insert_page_comment_tool,
        update_page_properties_tool,
        evaluate_title_tool
    ]
    
    prompt = create_notion_agent_prompt()
    
    logger.info("Notion agent created successfully")
    return create_react_agent(model=llm, tools=tools, prompt=prompt, name="notion_agent")

def run_notion_agent(query: str, history: list | None = None):
    history = history or []
    logger.info(f"Running notion agent with query: {query}")
    agent = create_notion_agent()
    result = agent.invoke({"messages": [HumanMessage(content=query)] + history})
    import pprint
    pprint.pprint(result, indent=4)
    logger.info("Notion agent execution completed")
    return result["messages"][-1].content
# Create the Notion agent instance
notion_agent = create_notion_agent()


# Example usage
if __name__ == "__main__":
    # Notion Agent 테스트 스크립트
    print("Notion Agent 테스트 시작")
    print("=" * 50)
    
    # 다양한 테스트 쿼리 준비
    test_queries = [
        "page with id 1e9ff0df28478038a184fe3371797f96에 title을 평가한 후 평가내용을 댓글로 남겨줘",
        # "Get all blocks from the page with id 1e9ff0df28478038a184fe3371797f96",
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
