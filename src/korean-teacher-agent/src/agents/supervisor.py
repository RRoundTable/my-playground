"""
Supervisor Agent using LangGraph Supervisor

This agent coordinates between the Notion agent and Title agent to manage workflows.
"""

from typing import Dict, List, Optional, Union, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv
from src.agents.notion_agent import notion_agent
from src.agents.title_agent import title_agent

# Load environment variables
load_dotenv()

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.7
)

# Create the supervisor workflow
workflow = create_supervisor(
    [notion_agent, title_agent],
    model=llm,
    supervisor_name="korean_teacher_supervisor",
    prompt=(
        "You are a team supervisor managing specialized agents:\n"
        "- title_agent: Evaluates titles and provides feedback\n"
        "- notion_agent: Interacts with Notion pages, blocks, and comments\n\n"
        "Always use title_agent to evaluate titles.\n"
        "Always use notion_agent for Notion-related tasks like fetching pages, blocks, or adding comments.\n"
        "Your job is to route tasks to the appropriate agent based on the user's query.\n"
        "For title evaluation or feedback, use title_agent.\n"
        "For Notion interactions, use notion_agent.\n"
        "Please respond in Korean language."
    )
    
)

# Compile the workflow
app = workflow.compile()

def run_supervisor_agent(
    query: str,
    chat_history: Optional[List[Union[HumanMessage, AIMessage]]] = None,
    agent_scratchpad: Optional[List[Union[HumanMessage, AIMessage]]] = None
) -> Dict:
    """
    Run the supervisor agent to coordinate between Notion and Title agents.
    
    Args:
        query (str): The user's query
        chat_history (Optional[List[Union[HumanMessage, AIMessage]]]): Previous conversation history
        agent_scratchpad (Optional[List[Union[HumanMessage, AIMessage]]]): Agent's working memory
        
    Returns:
        Dict: Results from the appropriate agent
    """
    # Convert query to message format
    messages = [{"role": "user", "content": query}]
    
    # Add chat history if exists
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})

    result = app.invoke({
        "messages": messages,
        "is_last_step": False,
        "remaining_steps": 5
    })
    
    return {
        "messages": result["messages"],
        "agent_used": result.get("active_agent", "unknown"),
        "result": result.get("output", "")
    }

# Example usage
if __name__ == "__main__":
    # Interactive CLI testing
    import sys
    
    print("Korean Teacher Agent CLI - Type 'exit' to quit")
    print("="*50)
    
    chat_history = []
    
    while True:
        try:
            user_input = input("\nEnter your query: ")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Exiting the Korean Teacher Agent CLI. Goodbye!")
                sys.exit(0)
                
            result = run_supervisor_agent(user_input, chat_history)
            
            print("\n" + "="*50)
            print(f"Agent Used: {result['agent_used']}")
            print("="*50)
            
            # Immediately display the response
            if 'messages' in result and result['messages']:
                last_message = result['messages'][-1]
                if hasattr(last_message, 'content'):
                    print(f"Response: {last_message.content}")
                elif isinstance(last_message, dict) and 'content' in last_message:
                    print(f"Response: {last_message['content']}")
            print("="*50)
            
            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            if result['messages']:
                last_message = result['messages'][-1]
                if hasattr(last_message, 'content'):
                    chat_history.append(AIMessage(content=last_message.content))
                elif isinstance(last_message, dict) and 'content' in last_message:
                    chat_history.append(AIMessage(content=last_message['content']))
                
        except KeyboardInterrupt:
            print("\nExiting the Korean Teacher Agent CLI. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {str(e)}")
