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
    
    Args:
        message: The title and content to evaluate
        
    Returns:
        str: Detailed evaluation in Korean
    """
    logger.info(f"Evaluating title with message: {message[:50]}...")
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

def run_title_agent(
    title: str,
    content: str,
    chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    agent_scratchpad: Optional[List[Union[HumanMessage, AIMessage]]] = None
) -> str:
    """Run the title agent to evaluate a title.
    
    Args:
        title (str): The title to evaluate
        content (str): The content to compare against
        chat_history (Optional[List]): Previous conversation history
        agent_scratchpad (Optional[List]): Agent's working memory
        
    Returns:
        str: The evaluation result
    """
    logger.info(f"Running title agent for title: '{title[:30]}...'")
    if chat_history is None:
        chat_history = []
        
    agent = create_title_agent()
    logger.info("Invoking title agent")
    result = agent.invoke({
        "input": f"제목 '{title}'이 다음 내용에 얼마나 잘 맞는지 평가해주세요:\n\n{content}",
        "chat_history": chat_history,
        "agent_scratchpad": agent_scratchpad
    })
    logger.info("Title agent evaluation completed")

    # Extract the final answer from the agent's response
    return result["messages"][-1].content

# Create the title agent instance
logger.info("Initializing title agent")
title_agent = create_title_agent()

# Example usage
if __name__ == "__main__":
    # Configure logging for direct script execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info("Title Agent 테스트 시작")
    print("Title Agent 테스트 시작")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            "title": "한국어 학습의 효과적인 방법",
            "content": """
            한국어를 배우는 외국인들을 위한 효과적인 학습 방법을 소개합니다.
            듣기, 말하기, 읽기, 쓰기의 균형 잡힌 학습이 중요하며,
            실제 한국인과의 대화 기회를 많이 가지는 것이 도움이 됩니다.
            또한 K-pop과 한국 드라마를 통한 문화 학습도 언어 습득에 큰 도움이 됩니다.
            체계적인 문법 학습과 함께 실생활에서 사용되는 표현을 익히는 것이 핵심입니다.
            """
        },
        {
            "title": "K-pop으로 배우는 한국어 - BTS 'Dynamite' 가사 분석",
            "content": """
            세계적으로 인기 있는 BTS의 'Dynamite' 가사를 통해 한국어를 배워봅시다.
            이 노래에 사용된 한국어 표현들을 하나씩 분석하고,
            실생활에서 어떻게 활용할 수 있는지 알아보겠습니다.
            특히 K-pop 팬들이 자주 사용하는 표현들을 중점적으로 다룹니다.
            """
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"Running test case #{i}")
        print(f"\n테스트 케이스 #{i}")
        print("-" * 50)
        print(f"제목: {test_case['title']}")
        print(f"\n내용: {test_case['content']}")
        
        try:
            result = run_title_agent(
                title=test_case['title'],
                content=test_case['content']
            )
            logger.info(f"Test case #{i} completed successfully")
            print("\n평가 결과:")
            print(result)
        except Exception as e:
            error_msg = f"Error in test case #{i}: {str(e)}"
            logger.error(error_msg)
            print(f"\n오류 발생: {str(e)}")
        
        print("-" * 50)
    
    logger.info("Title Agent 테스트 완료")
    print("\nTitle Agent 테스트 완료")
    print("=" * 50)