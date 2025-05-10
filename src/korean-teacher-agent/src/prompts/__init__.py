"""
Prompt management module for Korean Teacher Agent.
This module contains all the prompts used by different agents.
"""

from .notion_agent_prompts import create_notion_agent_prompt
from .title_agent_prompts import (
    create_title_evaluation_prompt,
    create_title_agent_prompt,
)

from .prompt_manager import prompt_manager
__all__ = [
    'create_notion_agent_prompt',
    'create_title_evaluation_prompt',
    'create_title_agent_prompt',
    'prompt_manager',
] 