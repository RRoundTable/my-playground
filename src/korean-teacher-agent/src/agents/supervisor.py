"""
Supervisor Agent using LangGraph Supervisor

This agent coordinates between the Notion agent and Title agent to manage workflows.
"""

from typing import Dict, List, Optional, Union, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv
from src.agents.notion_agent import notion_agent
from src.agents.title_agent import title_agent
from src.prompts import create_supervisor_prompt

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
    prompt=create_supervisor_prompt()
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
    # Initialize messages list
    messages = []
    
    # Add chat history if exists
    if chat_history:
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, ToolMessage):
                messages.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id})
    
    # Add the current query at the end
    messages.append({"role": "user", "content": query})

    result = app.invoke({
        "messages": messages,
        "is_last_step": False,
        "remaining_steps": 5
    })
    
    return {
        "messages": result["messages"],
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
