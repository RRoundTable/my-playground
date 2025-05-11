"""
Title evaluation tools for the Korean Teacher Agent.

This module provides tools for evaluating YouTube video titles for Korean language learning content.
"""

from typing import Dict, List, Optional, Union
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
import logging
from src.prompts import prompt_manager

# Configure logging
logger = logging.getLogger(__name__)

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
    prompt = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt")

    llm = ChatOpenAI(model="gpt-4.1-nano")
    # Format the prompt with title and content
    
    logger.debug(f"Formatted prompt created for title evaluation")
    
    # Get evaluation from LLM
    logger.info("Invoking LLM for title evaluation")
    prompt = prompt.format().messages
    prompt.append(HumanMessage(content=message))
    response = llm.invoke(prompt)
    logger.info("Received evaluation response from LLM")
    logger.info(f"Evaluation response: {response.content}")
    return response.content

# Export all tools
title_tools = [
    evaluate_title_tool,
] 