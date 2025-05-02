from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the state type
class AgentState(TypedDict):
    title: str
    content: str
    evaluation: str

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4-turbo-preview",
    temperature=0.7
)

# Define the evaluation prompt
evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert title evaluator. Your task is to evaluate how well the given title matches the content.
    Consider the following aspects:
    1. Relevance to the content
    2. Clarity and conciseness
    3. Engagement and appeal
    4. SEO-friendliness
    
    Provide a detailed evaluation and suggest improvements if needed."""),
    ("human", """
    Title: {title}
    
    Content: {content}
    
    Please evaluate this title and provide your analysis.
    """)
])

# Define the evaluation node
def evaluate_title(state: AgentState) -> AgentState:
    # Generate evaluation
    chain = evaluation_prompt | llm
    response = chain.invoke({
        "title": state["title"],
        "content": state["content"]
    })
    
    # Update state with evaluation
    state["evaluation"] = response.content
    return state

# Create the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("evaluate", evaluate_title)

# Add edges
workflow.add_edge("evaluate", END)
workflow.set_entry_point("evaluate")  # Set the entry point

# Compile the graph
app = workflow.compile()

def evaluate_title_agent(title: str, content: str) -> str:
    """
    Evaluate a title based on its content using the agent.
    
    Args:
        title (str): The title to evaluate
        content (str): The content to compare against
        
    Returns:
        str: The evaluation result
    """
    result = app.invoke({
        "title": title,
        "content": content,
        "evaluation": ""
    })
    return result["evaluation"]

# Test section
if __name__ == "__main__":
    # Test case 1
    test_title = "The Future of Artificial Intelligence"
    test_content = """
    Artificial Intelligence is rapidly transforming various industries. 
    From healthcare to finance, AI applications are becoming increasingly sophisticated.
    Machine learning algorithms are now capable of processing vast amounts of data
    and making complex decisions. The integration of AI in daily life continues to grow,
    raising important questions about ethics and regulation.
    """
    
    print("Test Case 1:")
    print("Title:", test_title)
    print("\nContent:", test_content)
    print("\nEvaluation:")
    print(evaluate_title_agent(test_title, test_content))
    
    # Test case 2
    test_title2 = "Cooking Pasta"
    test_content2 = """
    Quantum computing represents a revolutionary approach to computation.
    Unlike classical computers that use bits, quantum computers use qubits
    which can exist in multiple states simultaneously. This enables them
    to solve certain problems much faster than traditional computers.
    """
    
    print("\nTest Case 2:")
    print("Title:", test_title2)
    print("\nContent:", test_content2)
    print("\nEvaluation:")
    print(evaluate_title_agent(test_title2, test_content2)) 