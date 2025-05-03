from typing import Annotated, TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv
from src.prompts import create_title_evaluation_prompt

# Load environment variables
load_dotenv()

# Define the state type
class AgentState(TypedDict):
    """State for the title agent workflow."""
    title: str = "" # The title to evaluate
    content: str = ""  # The content to compare the title against
    evaluation: str = ""  # The evaluation result
    chat_history: List[Union[HumanMessage, AIMessage]] = []  # Chat history for context
    agent_scratchpad: List[Union[HumanMessage, AIMessage]] = []  # For agent's intermediate reasoning

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7
)

def create_title_agent():
    """
    Create and return a title evaluation agent.
    
    Returns:
        StateGraph: The compiled title evaluation agent
    """
    # Define the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("evaluate", evaluate_title)
    workflow.add_node("generate_response", generate_response)
    
    # Add edges
    workflow.add_edge("evaluate", "generate_response")
    workflow.add_edge("generate_response", END)
    
    # Set entry point
    workflow.set_entry_point("evaluate")
    
    # Compile the workflow
    return workflow.compile(name="title_agent")

def evaluate_title(state: AgentState) -> AgentState:
    """
    Evaluate the title based on the content.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: Updated state with evaluation
    """
    prompt = create_title_evaluation_prompt()
    formatted_prompt = prompt.format_prompt(
        title=state["title"],
        content=state["content"],
        chat_history=state["chat_history"],
        agent_scratchpad=state["agent_scratchpad"]
    )
    response = llm.invoke(formatted_prompt.to_messages())
    state["evaluation"] = response.content
    return state

def generate_response(state: AgentState) -> AgentState:
    """
    Generate a response based on the evaluation.
    
    Args:
        state (AgentState): The current state
        
    Returns:
        AgentState: Updated state with response
    """
    return state

# Create the agent
title_agent = create_title_agent()

# Test section  
if __name__ == "__main__":
    # Test title agent
    test_title = "한국어 학습의 효과적인 방법"
    test_content = """
    한국어를 배우는 외국인들을 위한 효과적인 학습 방법을 소개합니다.
    듣기, 말하기, 읽기, 쓰기의 균형 잡힌 학습이 중요하며,
    실제 한국인과의 대화 기회를 많이 가지는 것이 도움이 됩니다.
    또한 K-pop과 한국 드라마를 통한 문화 학습도 언어 습득에 큰 도움이 됩니다.
    체계적인 문법 학습과 함께 실생활에서 사용되는 표현을 익히는 것이 핵심입니다.
    """
    
    print("Title Agent Test:")
    print("Title:", test_title)
    print("\nContent:", test_content)
    
    # Create input state
    input_state = AgentState(
        title=test_title,
        content=test_content,
        evaluation="",
        chat_history=[],
        agent_scratchpad=[]
    )
    
    # Run the title agent
    result = title_agent.invoke(input_state)
    
    print("\nEvaluation Result:")
    print(result["evaluation"])