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
from langchain_core.runnables import RunnableConfig

from src.tools import all_tools, tools_by_name
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

# Create a short-term memory cache for storing page titles and other information
class MemoryCache:
    def __init__(self):
        self.page_titles = {}  # page_id -> title
        self.page_contents = {}  # page_id -> content
        self.other_data = {}  # For any other data we want to cache
    
    def get_page_title(self, page_id):
        return self.page_titles.get(page_id)
    
    def set_page_title(self, page_id, title):
        self.page_titles[page_id] = title
        
    def get_page_content(self, page_id):
        return self.page_contents.get(page_id)
    
    def set_page_content(self, page_id, content):
        self.page_contents[page_id] = content
        
    def store_data(self, key, value):
        self.other_data[key] = value
        
    def get_data(self, key):
        return self.other_data.get(key)

# Initialize the global memory cache
memory_cache = MemoryCache()

# Define the state type for our ReAct Agent
class AgentState(TypedDict):
    """The state of the notion agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memory: dict  # Add memory to agent state

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
            
            # Store page title in memory if applicable
            if tool_name == 'get_page' and 'properties' in result and 'title' in result['properties']:
                page_id = tool_args.get('page_id')
                page_title = result['properties']['title']
                memory_cache.set_page_title(page_id, page_title)
                logger.info(f"Stored page title for page_id {page_id}: {page_title}")
            
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
    model_with_tools = llm.bind_tools(all_tools)
    
    # Get the system prompt as a string
    system_prompt = prompt_manager.get_prompt("korean-youtube-planner").format()
    system_prompt_text = system_prompt.messages[0]["content"]
    
    # Add memory context to system prompt if available
    memory_context = ""
    # Add page titles from memory
    if memory_cache.page_titles:
        memory_context += "Page titles in memory:\n"
        for page_id, title in memory_cache.page_titles.items():
            memory_context += f"- Page ID: {page_id}, Title: {title}\n"
    
    if memory_context:
        system_prompt_text += f"\n\nMemory Context:\n{memory_context}"
    
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
    
    # Initial state with empty memory
    initial_state = {"messages": initial_messages, "memory": {}}
    
    # Invoke the agent
    result = agent.invoke(initial_state)
    
    # Extract and return the last message content
    logger.info("Notion agent execution completed")
    
    final_response = result["messages"][-1].content
    
    return final_response

# Create the Notion agent instance (for API use)
notion_agent = create_notion_agent()

# Utility functions to interact with memory cache
def get_page_title_from_cache(page_id):
    """Get page title from memory cache if available."""
    return memory_cache.get_page_title(page_id)

def store_page_title_in_cache(page_id, title):
    """Store page title in memory cache for future use."""
    memory_cache.set_page_title(page_id, title)

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
