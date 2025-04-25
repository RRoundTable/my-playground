import getpass
import os
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    """Schema for structured output of search queries."""
    search_query: Optional[str] = Field(
        None, 
        description="Query that is optimized for web search."
    )
    justification: Optional[str] = Field(
        None, 
        description="Why this query is relevant to the user's request."
    )


def setup_openai_api_key() -> None:
    """Set up OpenAI API key from environment or prompt user."""
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")


def create_llm(model: str = "gpt-4.1-mini") -> ChatOpenAI:
    """Create and return an OpenAI LLM instance."""
    return ChatOpenAI(model=model)


def multiply(a: int, b: int) -> int:
    """Multiplies two integers together.

    Args:
        a: The first integer to multiply
        b: The second integer to multiply

    Returns:
        The product of a and b
    """
    return a * b


def demonstrate_structured_output(llm: ChatOpenAI) -> None:
    """Demonstrate LLM with structured output capability."""
    structured_llm = llm.with_structured_output(SearchQuery)
    output = structured_llm.invoke(
        "How does Calcium CT score relate to high cholesterol?"
    )
    print("Structured Output Example:", output)


def demonstrate_tool_usage(llm: ChatOpenAI) -> None:
    """Demonstrate LLM with tool binding capability."""
    llm_with_tools = llm.bind_tools([multiply])
    msg = llm_with_tools.invoke("What is 2 times 3?")
    print("Tool Usage Example:", msg.tool_calls)


def main() -> None:
    """Main function to demonstrate LLM capabilities."""
    setup_openai_api_key()
    llm = create_llm()
    
    demonstrate_structured_output(llm)
    demonstrate_tool_usage(llm)


if __name__ == "__main__":
    main()