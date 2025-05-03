"""
Prompt management module for Korean Teacher Agent.
This module contains all the prompts used by different agents.
"""

from .notion_agent_prompts import NOTION_AGENT_PROMPT, create_notion_agent_prompt
from .title_agent_prompts import (
    TITLE_AGENT_EVALUATION_PROMPT,
    create_title_evaluation_prompt,
)

__all__ = [
    'NOTION_AGENT_PROMPT',
    'TITLE_AGENT_EVALUATION_PROMPT',
    'TITLE_AGENT_CHAT_PROMPT',
    'create_notion_agent_prompt',
    'create_title_evaluation_prompt',
    'create_title_chat_prompt',
] 