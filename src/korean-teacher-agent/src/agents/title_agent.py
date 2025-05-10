"""
Title Agent using LangChain and LangGraph

This agent evaluates YouTube video titles for Korean language learning content.
"""

from typing import Dict, List, Optional, Union      
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import logging
from src.prompts import prompt_manager

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


# Test script
if __name__ == "__main__":
    # Sample test data
    test_title_content = """
    제목: 한국어 초보자를 위한 10가지 기초 표현
    내용: 안녕하세요, 인사하기, 감사합니다, 죄송합니다 등의 기본 표현을 배우고 발음 연습을 합니다.
    실제 한국인들이 일상에서 자주 사용하는 표현들을 예문과 함께 설명합니다.
    """
    
    # Run the tool with test data
    print("테스트 실행 중...")
    evaluation_result = evaluate_title_tool(test_title_content)
    print("\n평가 결과:")
    print(evaluation_result)


