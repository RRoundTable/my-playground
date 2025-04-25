import getpass
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from graphviz import Digraph


def setup_openai_api_key() -> None:
    """Set up OpenAI API key from environment or prompt user."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


# Initialize LLM
setup_openai_api_key()
llm = ChatOpenAI(model="gpt-4")


# Graph state
class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str


# Nodes
def generate_joke(state: State) -> dict:
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content}


def check_punchline(state: State) -> str:
    """Gate function to check if the joke has a punchline"""
    # Simple check - does the joke contain "?" or "!"
    if "?" in state["joke"] or "!" in state["joke"]:
        return "Fail"
    return "Pass"


def improve_joke(state: State) -> dict:
    """Second LLM call to improve the joke"""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State) -> dict:
    """Third LLM call for final polish"""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}


def create_joke_workflow() -> StateGraph:
    """Create and return the joke generation workflow"""
    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("generate_joke", generate_joke)
    workflow.add_node("improve_joke", improve_joke)
    workflow.add_node("polish_joke", polish_joke)

    # Add edges to connect nodes
    workflow.add_edge(START, "generate_joke")
    workflow.add_conditional_edges(
        "generate_joke", check_punchline, {"Fail": "improve_joke", "Pass": END}
    )
    workflow.add_edge("improve_joke", "polish_joke")
    workflow.add_edge("polish_joke", END)

    return workflow


def visualize_workflow(workflow: StateGraph, output_path: str = "workflow") -> None:
    """Visualize the workflow and save it as an image file"""
    # Create a new directed graph
    dot = Digraph(comment='Joke Generation Workflow')
    dot.attr(rankdir='LR')  # Left to right layout

    # Add nodes
    dot.node('START', 'START', shape='circle')
    dot.node('END', 'END', shape='circle')
    dot.node('generate_joke', 'Generate Joke', shape='box')
    dot.node('improve_joke', 'Improve Joke', shape='box')
    dot.node('polish_joke', 'Polish Joke', shape='box')

    # Add edges
    dot.edge('START', 'generate_joke')
    dot.edge('generate_joke', 'improve_joke', label='No Punchline')
    dot.edge('generate_joke', 'END', label='Has Punchline')
    dot.edge('improve_joke', 'polish_joke')
    dot.edge('polish_joke', 'END')

    # Save the graph
    dot.render(output_path, format='png', cleanup=True)
    print(f"Workflow visualization saved as {output_path}")


def run_joke_workflow(topic: str) -> State:
    """Run the joke generation workflow with the given topic"""
    # Create and compile workflow
    workflow = create_joke_workflow()
    
    # Visualize the workflow
    visualize_workflow(workflow)
    
    chain = workflow.compile()

    # Run workflow
    state = chain.invoke({"topic": topic})
    
    # Print results
    print("Initial joke:")
    print(state["joke"])
    print("\n--- --- ---\n")
    
    if "improved_joke" in state:
        print("Improved joke:")
        print(state["improved_joke"])
        print("\n--- --- ---\n")
        print("Final joke:")
        print(state["final_joke"])
    else:
        print("Joke failed quality gate - no punchline detected!")
    
    return state


if __name__ == "__main__":
    # Example usage
    result = run_joke_workflow("cats") 