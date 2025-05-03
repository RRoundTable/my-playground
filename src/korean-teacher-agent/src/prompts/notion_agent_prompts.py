"""
Prompts for the Notion agent.
"""
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

NOTION_AGENT_PROMPT = """You are a helpful assistant that can interact with Notion.
You have access to tools that can fetch page information, blocks, and comments from Notion,
and can also insert new comments into blocks.
Use these tools to help users with their Notion-related tasks."""

def create_notion_agent_prompt() -> ChatPromptTemplate:
    """Create and return the ChatPromptTemplate for the Notion agent."""
    return ChatPromptTemplate.from_messages([
        ("system", NOTION_AGENT_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]) 