"""
Title Agent using LangChain and LangGraph

This agent evaluates YouTube video titles for Korean language learning content.
"""

from typing import Dict, List, Optional, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import logging
from src.prompts import create_title_evaluation_prompt, create_title_agent_prompt
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


@tool("evaluate_title")
def evaluate_title_tool(message: str) -> str:
    """Evaluate how well a title matches its content for Korean language learning videos.
    Check if the message contains enough information
    The message should be structured with both title and content information
    This tool requires both elements to perform a proper evaluation
    
    Args:
        message: The title and content to evaluate
        
    Returns:
        str: Detailed evaluation in Korean
    """
    logger.info(f"Evaluating title with message: {message}...")
    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
    prompt = create_title_evaluation_prompt()
    
    # Format the prompt with title and content
    formatted_prompt = prompt.format_prompt(
       input=message,
       chat_history=[],
       agent_scratchpad=[]
    )
    logger.debug(f"Formatted prompt created for title evaluation")
    
    # Get evaluation from LLM
    logger.info("Invoking LLM for title evaluation")
    response = llm.invoke(formatted_prompt.to_messages())
    logger.info("Received evaluation response from LLM")
    logger.info(f"Evaluation response: {response.content}")
    return response.content

def create_title_agent():
    """Create and return a React-based title evaluation agent.
    
    Returns:
        Agent: The compiled title evaluation agent
    """
    logger.info("Creating title evaluation agent")
    llm = ChatOpenAI(temperature=0, model="gpt-4.1-nano")
    
    tools = [evaluate_title_tool]
    
    prompt = create_title_agent_prompt()
    logger.debug("Title agent prompt created")

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=prompt,
        name="title_agent"
    )
    logger.info("Title agent created successfully")
    return agent