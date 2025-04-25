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
    story: str
    poem: str
    combined_output: str


# Nodes
def call_llm_1(state: State) -> dict:
    """First LLM call to generate initial joke"""
    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}


def call_llm_2(state: State) -> dict:
    """Second LLM call to generate story"""
    msg = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": msg.content}


def call_llm_3(state: State) -> dict:
    """Third LLM call to generate poem"""
    msg = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: State) -> dict:
    """Combine the joke, story, and poem into a single output"""
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


def create_parallel_workflow() -> StateGraph:
    """Create and return the parallel workflow"""
    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("call_llm_1", call_llm_1)
    workflow.add_node("call_llm_2", call_llm_2)
    workflow.add_node("call_llm_3", call_llm_3)
    workflow.add_node("aggregator", aggregator)

    # Add edges to connect nodes
    workflow.add_edge(START, "call_llm_1")
    workflow.add_edge(START, "call_llm_2")
    workflow.add_edge(START, "call_llm_3")
    workflow.add_edge("call_llm_1", "aggregator")
    workflow.add_edge("call_llm_2", "aggregator")
    workflow.add_edge("call_llm_3", "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow


def visualize_workflow(workflow: StateGraph, output_path: str = "parallel_workflow") -> None:
    """Visualize the workflow and save it as an image file"""
    # Create a new directed graph
    dot = Digraph(comment='Parallel Content Generation Workflow')
    dot.attr(rankdir='TB')  # Top to bottom layout

    # Add nodes
    dot.node('START', 'START', shape='circle')
    dot.node('END', 'END', shape='circle')
    dot.node('call_llm_1', 'Generate Joke', shape='box')
    dot.node('call_llm_2', 'Generate Story', shape='box')
    dot.node('call_llm_3', 'Generate Poem', shape='box')
    dot.node('aggregator', 'Combine Output', shape='box')

    # Add edges
    dot.edge('START', 'call_llm_1')
    dot.edge('START', 'call_llm_2')
    dot.edge('START', 'call_llm_3')
    dot.edge('call_llm_1', 'aggregator')
    dot.edge('call_llm_2', 'aggregator')
    dot.edge('call_llm_3', 'aggregator')
    dot.edge('aggregator', 'END')

    # Save the graph
    dot.render(output_path, format='png', cleanup=True)
    print(f"Workflow visualization saved as {output_path}.png")


def run_parallel_workflow(topic: str) -> State:
    """Run the parallel workflow with the given topic"""
    # Create workflow
    workflow = create_parallel_workflow()
    
    # Visualize the workflow
    visualize_workflow(workflow)
    
    # Compile and run workflow
    compiled_workflow = workflow.compile()
    state = compiled_workflow.invoke({"topic": topic})
    
    # Print results
    print("\nGenerated Content:")
    print(state["combined_output"])
    
    return state


if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run parallel content generation workflow')
    parser.add_argument('--topic', type=str, default='cats', help='Topic for content generation')
    args = parser.parse_args()
    
    # Run workflow
    result = run_parallel_workflow(args.topic) 