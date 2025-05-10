"""
Notion Agent using LangChain and LangGraph

This agent provides tools to interact with Notion API using LangChain and LangGraph.
"""

from typing import Dict, List, Optional, Type, Any, Annotated, Sequence, TypedDict
import os
import logging
import json
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import requests
from src.prompts import create_notion_agent_prompt
from src.clients.notion_client import NotionAPIClient, NotionAPIError
from src.agents.title_agent import evaluate_title_tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
# Import Phoenix OpenTelemetry
from phoenix.otel import register, TracerProvider

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
    headers={"Authorization": f"Bearer {phoenix_secret}"} if phoenix_secret else {}
)   

# tracer 생성
tracer = tracer_provider.get_tracer("notion_agent")

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
    with tracer.start_as_current_span("get_notion_page") as span:
        span.set_attribute("page_id", page_id)
        logger.info(f"Tool used: get_notion_page with page_id: {page_id}")
        try:
            result = notion_client.get_page(page_id)
            logger.info(f"Tool output: get_notion_page returned data of length: {len(str(result))}")
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to fetch page {page_id}: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to fetch page {page_id}: {str(e)}")

@tool("get_notion_page_paragraph_text_blocks")
def get_page_paragraph_text_blocks_tool(page_id: str) -> List[Dict]:
    """Fetch all blocks from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch blocks from
        
    Returns:
        List[Dict]: List of blocks from the Notion page
    """
    with tracer.start_as_current_span("get_notion_page_paragraph_text_blocks") as span:
        span.set_attribute("page_id", page_id)
        logger.info(f"Tool used: get_notion_page_paragraph_text_blocks with page_id: {page_id}")
        try:
            result = notion_client.get_paragraph_text_blocks(page_id)
            logger.info(f"Tool output: get_notion_page_paragraph_text_blocks returned {len(result)} blocks")
            span.set_attribute("blocks_count", len(result))
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to fetch blocks for page {page_id}: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to fetch blocks for page {page_id}: {str(e)}")

@tool("get_notion_page_comment_content_blocks")
def get_page_comment_content_blocks_tool(page_id: str) -> List[Dict]:
    """Fetch all comments from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion page
    """
    with tracer.start_as_current_span("get_notion_page_comment_content_blocks") as span:
        span.set_attribute("page_id", page_id)
        logger.info(f"Tool used: get_notion_page_comment_content_blocks with page_id: {page_id}")
        try:
            result = notion_client.get_comment_content_blocks(block_id=page_id)
            logger.info(f"Tool output: get_notion_page_comment_content_blocks returned {len(result)} comments")
            span.set_attribute("comments_count", len(result))
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to fetch comments for page {page_id}: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to fetch comments for page {page_id}: {str(e)}")

@tool("get_notion_block_comments")
def get_block_comments_tool(block_id: str) -> List[Dict]:
    """Fetch all comments from a specific Notion block.
    
    Args:
        block_id (str): The ID of the Notion block to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion block
    """
    with tracer.start_as_current_span("get_notion_block_comments") as span:
        span.set_attribute("block_id", block_id)
        logger.info(f"Tool used: get_notion_block_comments with block_id: {block_id}")
        try:
            result = notion_client.get_comments(block_id=block_id)
            logger.info(f"Tool output: get_notion_block_comments returned {len(result)} comments")
            span.set_attribute("comments_count", len(result))
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to fetch comments for block {block_id}: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to fetch comments for block {block_id}: {str(e)}")

@tool("insert_notion_comment")
def insert_comment_tool(tool_input: NotionCommentInput) -> Dict:
    """Insert a comment into a Notion block.
    
    Args:
        tool_input (NotionCommentInput): Input containing block_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    with tracer.start_as_current_span("insert_notion_comment") as span:
        span.set_attribute("block_id", tool_input.block_id)
        span.set_attribute("text_length", len(tool_input.text))
        logger.info(f"Tool used: insert_notion_comment for block_id: {tool_input.block_id}")
        logger.info(f"Tool input: text length: {len(tool_input.text)}")
        try:
            result = notion_client.insert_comment(
                text=tool_input.text,
                block_id=tool_input.block_id
            )
            logger.info(f"Tool output: insert_notion_comment successfully created comment")
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to insert comment: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to insert comment: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid input: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Invalid input: {str(e)}")

@tool("get_notion_page_title")
def get_page_title_tool(page_id: str) -> str:
    """Get the title of a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page
        
    Returns:
        str: The title of the Notion page
    """
    with tracer.start_as_current_span("get_notion_page_title") as span:
        span.set_attribute("page_id", page_id)
        logger.info(f"Tool used: get_notion_page_title with page_id: {page_id}")
        try:
            page_data = notion_client.get_page(page_id)
            title_property = page_data.get("properties", {}).get("Title", {})
            if not title_property:
                logger.error(f"Title property not found in page {page_id}")
                span.set_attribute("success", False)
                span.set_attribute("error", "Title property not found in page")
                raise Exception("Title property not found in page")
                
            title_text = title_property.get("title", [])
            if not title_text:
                logger.info("Tool output: get_notion_page_title returned empty title")
                span.set_attribute("success", True)
                span.set_attribute("empty_title", True)
                return ""
            result = title_text[0].get("plain_text", "")
            logger.info(f"Tool output: get_notion_page_title returned title: '{result}'")
            span.set_attribute("success", True)
            span.set_attribute("title", result)
            return result
            
        except NotionAPIError as e:
            logger.error(f"Failed to fetch page title for {page_id}: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to fetch page title for {page_id}: {str(e)}")

@tool("insert_notion_page_comment")
def insert_page_comment_tool(tool_input: NotionPageCommentInput) -> Dict:
    """Insert a comment into a Notion page.
    
    Args:
        tool_input (NotionPageCommentInput): Input containing page_id and text
        
    Returns:
        Dict: The created comment data from Notion API
    """
    with tracer.start_as_current_span("insert_notion_page_comment") as span:
        span.set_attribute("page_id", tool_input.page_id)
        span.set_attribute("text_length", len(tool_input.text))
        logger.info(f"Tool used: insert_notion_page_comment for page_id: {tool_input.page_id}")
        logger.info(f"Tool input: text length: {len(tool_input.text)}")
        try:
            result = notion_client.insert_comment(
                text=tool_input.text,
                page_id=tool_input.page_id
            )
            logger.info(f"Tool output: insert_notion_page_comment successfully created comment")
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to insert page comment: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to insert page comment: {str(e)}")
        except ValueError as e:
            logger.error(f"Invalid input: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Invalid input: {str(e)}")


@tool("update_notion_page_properties")
def update_page_properties_tool(tool_input: NotionUpdatePagePropertiesInput) -> Dict:
    """Update properties of a Notion page.
    
    Args:
        tool_input (NotionUpdatePagePropertiesInput): Input containing page_id and properties
        
    Returns:
        Dict: The updated page data from Notion API
    """
    with tracer.start_as_current_span("update_notion_page_properties") as span:
        span.set_attribute("page_id", tool_input.page_id)
        span.set_attribute("properties_keys", str(list(tool_input.properties.keys())))
        logger.info(f"Tool used: update_notion_page_properties for page_id: {tool_input.page_id}")
        logger.info(f"Tool input: properties keys: {list(tool_input.properties.keys())}")
        try:
            result = notion_client.update_page_properties(
                page_id=tool_input.page_id,
                properties=tool_input.properties
            )
            logger.info(f"Tool output: update_notion_page_properties successfully updated properties")
            span.set_attribute("success", True)
            return result
        except NotionAPIError as e:
            logger.error(f"Failed to update page properties: {str(e)}")
            span.set_attribute("success", False)
            span.set_attribute("error", str(e))
            raise Exception(f"Failed to update page properties: {str(e)}")

# Define our list of tools
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

# Create a mapping of tool names to tools for easier access
tools_by_name = {tool.name: tool for tool in tools}

# Define the node for handling tool calls
def tool_node(state: AgentState) -> Dict:
    """Execute tool calls from the AI's last message."""
    with tracer.start_as_current_span("tool_node") as span:
        logger.info("Executing tool node")
        outputs = []
        # Get the last message which should be from the AI with tool calls
        last_message = state["messages"][-1]
        
        tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else []
        span.set_attribute("tool_calls_count", len(tool_calls))
        
        # Process each tool call
        for tool_call in tool_calls:
            logger.info(f"Processing tool call: {tool_call['name']}")
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]
            
            with tracer.start_as_current_span(f"tool_execution_{tool_name}") as tool_span:
                tool_span.set_attribute("tool_name", tool_name)
                tool_span.set_attribute("tool_args", str(tool_args))
                
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
                    tool_span.set_attribute("success", True)
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    outputs.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            name=tool_name,
                            tool_call_id=tool_id
                        )
                    )
                    tool_span.set_attribute("success", False)
                    tool_span.set_attribute("error", str(e))
        
        return {"messages": outputs}

# Define the node that calls the model
def call_model(state: AgentState, config: RunnableConfig):
    """Call the LLM to process the current conversation state."""
    with tracer.start_as_current_span("call_model") as span:
        logger.info("Calling LLM model")
        
        # Create the LLM
        llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
        model_with_tools = llm.bind_tools(tools)
        
        # Get the system prompt as a string
        system_prompt_text = create_notion_agent_prompt()
        
        # Create a system message with the prompt
        messages = [SystemMessage(content=system_prompt_text)]
        
        # Add conversation history
        messages.extend(state["messages"])
        
        span.set_attribute("messages_count", len(messages))
        
        # Invoke the model
        response = model_with_tools.invoke(messages, config)
        logger.info("Model response received")
        
        # Capture if the model response contains tool calls
        has_tool_calls = hasattr(response, "tool_calls") and len(response.tool_calls) > 0
        span.set_attribute("has_tool_calls", has_tool_calls)
        if has_tool_calls:
            span.set_attribute("tool_calls_count", len(response.tool_calls))
            span.set_attribute("tool_calls", str([tc["name"] for tc in response.tool_calls]))
        
        # Return the model's response
        return {"messages": [response]}

# Define the conditional edge function
def should_continue(state: AgentState) -> str:
    """Determine whether to continue with tool execution or end the conversation."""
    with tracer.start_as_current_span("should_continue") as span:
        logger.info("Checking if we should continue")
        
        # Get the last message
        last_message = state["messages"][-1]
        
        # If the last message has tool calls, continue to tools node
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
        span.set_attribute("has_tool_calls", has_tool_calls)
        
        if has_tool_calls:
            logger.info("Tool calls found, continuing to tools node")
            span.set_attribute("decision", "continue")
            return "continue"
        
        # If no tool calls, end the conversation
        logger.info("No tool calls found, ending the conversation")
        span.set_attribute("decision", "end")
        return "end"

def create_notion_agent():
    """Create and return a LangGraph-based ReAct agent."""
    with tracer.start_as_current_span("create_notion_agent") as span:
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
        span.set_attribute("success", True)
        return agent

def run_notion_agent(query: str, history: list | None = None):
    """Run the notion agent with a query and optional history."""
    with tracer.start_as_current_span("run_notion_agent") as span:
        span.set_attribute("query", query)
        span.set_attribute("has_history", history is not None)
        if history:
            span.set_attribute("history_length", len(history))
        
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
        span.set_attribute("response_length", len(final_response))
        span.set_attribute("success", True)
        
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
