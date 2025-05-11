"""
Tools registry for the Korean Teacher Agent.

This module aggregates all tools for the agent to use.
"""

from typing import Dict, List
from langchain_core.tools import BaseTool

from src.tools.notion_tools import notion_tools

# Aggregate all tools
all_tools: List[BaseTool] = [
    *notion_tools,
]

# Create a mapping of tool names to tools for easier access
tools_by_name: Dict[str, BaseTool] = {tool.name: tool for tool in all_tools} 