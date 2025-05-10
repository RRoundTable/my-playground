"""
Notion Agent using LangChain and LangGraph

This agent provides tools to interact with Notion API using LangChain and LangGraph.
"""

from typing import Dict, List, Optional, Type, Any, Annotated, Sequence, TypedDict
import os
import logging
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.clients.notion_client import NotionAPIClient, NotionAPIError
from src.agents.title_agent import evaluate_title_tool
from src.prompts import prompt_manager

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from phoenix.otel import register

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('notion_agent')

# Initialize OpenTelemetry with Phoenix
# Phoenix server 도메인으로 환경 변수 설정

phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
phoenix_secret = os.getenv("PHOENIX_API_KEY", "")


logger.info(f"Phoenix endpoint: {phoenix_endpoint}")
logger.info(f"Phoenix API key configured: {bool(phoenix_secret)}")

 
# register 함수로 설정 초기화 (API 키가 있으면 자동으로 Authorization 헤더 추가)
tracer_provider = register(
    project_name="notion-agent",
    protocol="grpc",
    endpoint=phoenix_endpoint,
    headers={"Authorization": f"Bearer {phoenix_secret}"} if phoenix_secret else {},
    auto_instrument=True
)   

# Initialize the API client
notion_client = NotionAPIClient()

# Define the state type for our ReAct Agent
class AgentState(TypedDict):
    """The state of the notion agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

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
def insert_comment_tool(block_id: str, text: str) -> Dict:
    """Insert a comment into a Notion block.
    
    Args:
        block_id (str): The ID of the Notion block to comment on
        text (str): The text content of the comment
        
    Returns:
        Dict: The created comment data from Notion API
    """
    logger.info(f"Tool used: insert_notion_comment for block_id: {block_id}")
    logger.info(f"Tool input: text length: {len(text)}")
    try:
        result = notion_client.insert_comment(
            text=text,
            block_id=block_id
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
def insert_page_comment_tool(page_id: str, text: str) -> Dict:
    """Insert a comment into a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to comment on
        text (str): The text content of the comment
        
    Returns:
        Dict: The created comment data from Notion API
    """
    logger.info(f"Tool used: insert_notion_page_comment for page_id: {page_id}")
    logger.info(f"Tool input: text length: {len(text)}")
    try:
        result = notion_client.insert_comment(
            text=text,
            page_id=page_id
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
def update_page_properties_tool(page_id: str, properties: Dict) -> Dict:
    """Update properties of a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page
        properties (Dict): Dictionary of properties to update
        
    Returns:
        Dict: The updated page data from Notion API
    """
    logger.info(f"Tool used: update_notion_page_properties for page_id: {page_id}")
    logger.info(f"Tool input: properties keys: {list(properties.keys())}")
    try:
        result = notion_client.update_page_properties(
            page_id=page_id,
            properties=properties
        )
        logger.info(f"Tool output: update_notion_page_properties successfully updated properties")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to update page properties: {str(e)}")
        raise Exception(f"Failed to update page properties: {str(e)}")

# Define our list of tools
tools = [
    get_page_paragraph_text_blocks_tool,
    get_page_comment_content_blocks_tool,
    get_block_comments_tool,
    insert_comment_tool,
    get_page_title_tool,
    insert_page_comment_tool,
    update_page_properties_tool,
    evaluate_title_tool
]

# Create a mapping of tool names to tools for easier access
tools_by_name = {tool.name: tool for tool in tools}

# Define the node for handling tool calls
def tool_node(state: AgentState) -> Dict:
    """Execute tool calls from the AI's last message."""
    logger.info("Executing tool node")
    outputs = []
    # Get the last message which should be from the AI with tool calls
    last_message = state["messages"][-1]
    
    tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []
    
    # Process each tool call
    for tool_call in tool_calls:
        logger.info(f"Processing tool call: {tool_call['name']}")
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
            
        # Get the tool by name and invoke it
        try:
            result = tools_by_name[tool_name].invoke(tool_args)
            
            outputs.append(
                ToolMessage(
                    content=json.dumps(result) if not isinstance(result, str) else result,
                    name=tool_name,
                    tool_call_id=tool_id
                )
            )
            logger.info(f"Tool {tool_name} executed successfully")
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            
            outputs.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_id
                )
            )
    
    return {"messages": outputs}

# Define the node that calls the model
def call_model(state: AgentState, config: RunnableConfig):
    """Call the LLM to process the current conversation state."""
    logger.info("Calling LLM model")
    
    # Create the LLM
    llm = ChatOpenAI(model="gpt-4.1-nano")
    model_with_tools = llm.bind_tools(tools)
    
    # Get the system prompt as a string
    system_prompt = prompt_manager.get_prompt("korean-youtube-planner").format()
    system_prompt_text = system_prompt.messages[0]["content"]
    
    
    # Create a system message with the prompt
    messages = [SystemMessage(content=system_prompt_text)]
    # Add conversation history
    messages.extend(state["messages"])
    
    
    # Invoke the model
    response = model_with_tools.invoke(messages, config)
    logger.info("Model response received")
    
    # Return the model's response
    return {"messages": [response]}

# Define the conditional edge function
def should_continue(state: AgentState) -> str:
    """Determine whether to continue with tool execution or end the conversation."""
    logger.info("Checking if we should continue")
    
    # Get the last message
    last_message = state["messages"][-1]
    
    # If the last message has tool calls, continue to tools node
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    
    if has_tool_calls:
        logger.info("Tool calls found, continuing to tools node")
        return "continue"
    
    # If no tool calls, end the conversation
    logger.info("No tool calls found, ending the conversation")
    return "end"

def create_notion_agent():
    """Create and return a LangGraph-based ReAct agent."""
    logger.info("Creating notion agent")
    
    # Define the agent graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    agent = workflow.compile()
    
    logger.info("Notion agent created successfully")
    return agent

def run_notion_agent(query: str, history: list | None = None):
    """Run the notion agent with a query and optional history."""
    
    history = history or []
    logger.info(f"Running notion agent with query: {query}")
    
    # Create the agent
    agent = create_notion_agent()
    
    # Convert history to BaseMessage objects if needed
    if history and not isinstance(history[0], BaseMessage):
        history_messages = []
        for msg in history:
            if isinstance(msg, tuple) and len(msg) == 2:
                role, content = msg
                if role == "user" or role == "human":
                    history_messages.append(HumanMessage(content=content))
                elif role == "assistant" or role == "ai":
                    history_messages.append(AIMessage(content=content))
            elif isinstance(msg, dict) and "role" in msg and "content" in msg:
                if msg["role"] == "user" or msg["role"] == "human":
                    history_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant" or msg["role"] == "ai":
                    history_messages.append(AIMessage(content=msg["content"]))
        history = history_messages
    
    # Prepare the initial message state
    initial_messages = history + [HumanMessage(content=query)]
    
    
    # Invoke the agent
    result = agent.invoke({"messages": initial_messages})
    
    # Extract and return the last message content
    logger.info("Notion agent execution completed")
    
    final_response = result["messages"][-1].content
    
    return final_response

# Create the Notion agent instance (for API use)
notion_agent = create_notion_agent()

# Example usage
if __name__ == "__main__":
    # Notion Agent 테스트 스크립트
    print("Notion Agent 테스트 시작")
    print("=" * 50)
    
    # 다양한 테스트 쿼리 준비
    test_queries = [
        "page id 1e9ff0df28478038a184fe3371797f96에 title을 평가한 후 평가내용을 노션 댓글로 남겨줘",
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
