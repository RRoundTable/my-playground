"""
Notion API tools for the Korean Teacher Agent.

This module provides tools to interact with Notion API.
"""

from typing import Dict, List
import os
import logging
from langchain_core.tools import tool

from src.clients.notion_client import NotionAPIClient, NotionAPIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('notion_tools')

# Initialize the API client
notion_client = NotionAPIClient()

def get_notion_headers():
    """Get Notion API headers."""
    notion_token = os.getenv("NOTION_TOKEN")
    return {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

def get_base_url():
    """Get Notion API base URL."""
    return "https://api.notion.com/v1"

@tool("get_notion_page")
def get_page_tool(page_id: str) -> Dict:
    """Fetch a Notion page by its ID.
    Used to query notion page properties.
    
    Args:
        page_id (str): The ID of the Notion page to fetch
        
    Returns:
        Dict: The page data from Notion API
    """
    logger.info(f"Tool used: get_notion_page with page_id: {page_id}")
    try:
        result = notion_client.get_page(page_id)
        logger.info(f"Tool output: get_notion_page returned data of length: {len(str(result))}")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch page {page_id}: {str(e)}")

@tool("get_notion_page_paragraph_text_blocks")
def get_page_paragraph_text_blocks_tool(page_id: str) -> List[Dict]:
    """Fetch all blocks from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch blocks from
        
    Returns:
        List[Dict]: List of blocks from the Notion page
    """
    logger.info(f"Tool used: get_notion_page_paragraph_text_blocks with page_id: {page_id}")
    try:
        result = notion_client.get_paragraph_text_blocks(page_id)
        logger.info(f"Tool output: get_notion_page_paragraph_text_blocks returned {len(result)} blocks")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch blocks for page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch blocks for page {page_id}: {str(e)}")

@tool("get_notion_page_comment_content_blocks")
def get_page_comment_content_blocks_tool(page_id: str) -> List[Dict]:
    """Fetch all comments from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion page
    """
    logger.info(f"Tool used: get_notion_page_comment_content_blocks with page_id: {page_id}")
    try:
        result = notion_client.get_comment_content_blocks(block_id=page_id)
        logger.info(f"Tool output: get_notion_page_comment_content_blocks returned {len(result)} comments")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch comments for page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch comments for page {page_id}: {str(e)}")

@tool("get_notion_block_comments")
def get_block_comments_tool(block_id: str) -> List[Dict]:
    """Fetch all comments from a specific Notion block.
    
    Args:
        block_id (str): The ID of the Notion block to fetch comments from
        
    Returns:
        List[Dict]: List of comments from the Notion block
    """
    logger.info(f"Tool used: get_notion_block_comments with block_id: {block_id}")
    try:
        result = notion_client.get_comments(block_id=block_id)
        logger.info(f"Tool output: get_notion_block_comments returned {len(result)} comments")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch comments for block {block_id}: {str(e)}")
        raise Exception(f"Failed to fetch comments for block {block_id}: {str(e)}")

@tool("insert_notion_block_comment")
def insert_block_comment_tool(block_id: str, text: str) -> Dict:
    """Insert a comment into a Notion block.
    Used when leaving a comment on a specific block within a notion page.
    For example, when leaving a review of a specific block within a notion page.
    Args:
        block_id (str): The ID of the Notion block to comment on
        text (str): The text content of the comment
        
    Returns:
        Dict: The created comment data from Notion API
    """
    logger.info(f"Tool used: insert_notion_comment for block_id: {block_id}")
    logger.info(f"Tool input: text length: {len(text)}")
    try:
        result = notion_client.insert_comment(
            text=text,
            block_id=block_id
        )
        logger.info(f"Tool output: insert_notion_comment successfully created comment")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to insert comment: {str(e)}")
        raise Exception(f"Failed to insert comment: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise Exception(f"Invalid input: {str(e)}")

@tool("get_notion_page_title")
def get_page_title_tool(page_id: str) -> str:
    """Get the title of a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page
        
    Returns:
        str: The title of the Notion page
    """
    logger.info(f"Tool used: get_notion_page_title with page_id: {page_id}")
    try:
        page_data = notion_client.get_page(page_id)
        title_property = page_data.get("properties", {}).get("Title", {})
        if not title_property:
            logger.error(f"Title property not found in page {page_id}")
            raise Exception("Title property not found in page")
            
        title_text = title_property.get("title", [])
        if not title_text:
            logger.info("Tool output: get_notion_page_title returned empty title")
            return ""
        result = title_text[0].get("plain_text", "")
        logger.info(f"Tool output: get_notion_page_title returned title: '{result}'")
        return result
        
    except NotionAPIError as e:
        logger.error(f"Failed to fetch page title for {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch page title for {page_id}: {str(e)}")

@tool("insert_notion_page_comment")
def insert_page_comment_tool(page_id: str, text: str) -> Dict:
    """Insert a comment into a Notion page.
    Use this when leaving a comment on the entire page. For example, when leaving a review of the page title.

    Args:
        page_id (str): The ID of the Notion page to comment on
        text (str): The text content of the comment
        
    Returns:
        Dict: The created comment data from Notion API
    """
    logger.info(f"Tool used: insert_notion_page_comment for page_id: {page_id}")
    logger.info(f"Tool input: text length: {len(text)}")
    try:
        result = notion_client.insert_comment(
            text=text,
            page_id=page_id
        )
        logger.info(f"Tool output: insert_notion_page_comment successfully created comment")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to insert page comment: {str(e)}")
        raise Exception(f"Failed to insert page comment: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid input: {str(e)}")
        raise Exception(f"Invalid input: {str(e)}")

@tool("update_notion_page_properties")
def update_page_properties_tool(page_id: str, properties: Dict) -> Dict:
    """Update properties of a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page
        properties (Dict): Dictionary of properties to update
        
    Returns:
        Dict: The updated page data from Notion API
    """
    logger.info(f"Tool used: update_notion_page_properties for page_id: {page_id}")
    logger.info(f"Tool input: properties keys: {list(properties.keys())}")
    try:
        result = notion_client.update_page_properties(
            page_id=page_id,
            properties=properties
        )
        logger.info(f"Tool output: update_notion_page_properties successfully updated properties")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to update page properties: {str(e)}")
        raise Exception(f"Failed to update page properties: {str(e)}")
    
@tool("get_notion_page_blocks")
def get_page_blocks_tool(page_id: str) -> List[Dict]:
    """Get all blocks from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page

    Returns:
        List[Dict]: List of blocks from the Notion page
    """
    logger.info(f"Tool used: get_notion_page_blocks with page_id: {page_id}")
    try:
        result = notion_client.get_page_blocks(page_id)
        logger.info(f"Tool output: get_notion_page_blocks returned {len(result)} blocks")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch blocks for page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch blocks for page {page_id}: {str(e)}")
    
def _extract_plain_text_from_raw_block(block: Dict) -> str:
    """Extracts plain text content from a raw Notion block dictionary."""
    block_type = block.get("type")
    if not block_type or block_type not in block:
        return ""

    rich_text_list = []
    # Common block types that store rich_text directly under their type key
    if block_type in [
        "paragraph", "heading_1", "heading_2", "heading_3", 
        "quote", "callout", "bulleted_list_item", 
        "numbered_list_item", "to_do", "toggle"
    ]:
        rich_text_list = block[block_type].get("rich_text", [])
    # Add checks for other block types if they store text differently
    # For example, table cells might be block[block_type]['cells'][row_idx][col_idx]['rich_text'] - more complex

    plain_text_parts = []
    for item in rich_text_list:
        if item.get("type") == "text":
            plain_text_parts.append(item.get("plain_text", ""))
    
    return "".join(plain_text_parts).strip()

@tool("get_notion_page_blocks")
def get_notion_page_blocks_tool(page_id: str) -> List[Dict]:
    """
    Fetches all raw blocks from a Notion page.
    
    Args:
        page_id (str): The ID of the Notion page to fetch blocks from
        
    Returns:
        List[Dict]: List of raw block data from the Notion page
    """
    logger.info(f"Tool used: get_notion_page_blocks_tool with page_id: {page_id}")
    try:
        result = notion_client.get_page_blocks(page_id) 
        logger.info(f"Tool output: get_notion_page_blocks_tool returned {len(result)} raw blocks")
        return result
    except NotionAPIError as e:
        logger.error(f"Failed to fetch raw blocks for page {page_id}: {str(e)}")
        raise Exception(f"Failed to fetch raw blocks for page {page_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in get_notion_page_blocks_tool for page {page_id}: {str(e)}", exc_info=True)
        raise Exception(f"Unexpected error fetching raw blocks for page {page_id}: {str(e)}")

@tool("parse_notion_page_into_sections_tool")
def parse_notion_page_into_sections_tool(page_id: str) -> Dict[str, List[Dict]]:
    """
    Fetches all blocks from a Notion page, extracts their ID, type, and plain text,
    and then parses them into logical sections (title, thumbnail, intro, body, other)
    based on heading_1 blocks and their content.
    """
    logger.info(f"Tool used: parse_notion_page_into_sections for page_id: {page_id}")
    try:
        raw_blocks = notion_client.get_page_blocks(page_id)
        if not raw_blocks:
            logger.info(f"No blocks found for page {page_id}.")
            return {"title": [], "thumbnail": [], "intro": [], "body": [], "other": []}

        processed_blocks = []
        for block_data in raw_blocks:
            plain_text = _extract_plain_text_from_raw_block(block_data)
            processed_blocks.append({
                "id": block_data.get("id"),
                "type": block_data.get("type"),
                "plain_text": plain_text
            })

        sections: Dict[str, List[Dict]] = {
            "title": [],
            "thumbnail": [],
            "intro": [],
            "body": {"main_content": [], "1분_3분": [], "3분_6분": [], "6분_이후": [], "other_h2_content": []}, # Initialize body with sub-sections
            "other": []
        }
        current_section_key = "other" # Default section for blocks before any recognized heading
        current_body_subsection_key = "main_content" # Default for content directly under body H1

        for p_block in processed_blocks:
            is_heading_and_processed = False
            if p_block["type"] == "heading_1":
                heading_text = p_block["plain_text"].lower()
                current_body_subsection_key = "main_content" # Reset body subsection when H1 changes

                if "제목" in heading_text:
                    current_section_key = "title"
                elif "썸네일" in heading_text:
                    current_section_key = "thumbnail"
                elif "인트로" in heading_text or "초반 30초" in heading_text :
                    current_section_key = "intro"
                elif "본문" in heading_text:
                    current_section_key = "body"
                else:
                    current_section_key = "other"
                
                # Add H1 block to its section (or main_content of body)
                if current_section_key == "body":
                    sections[current_section_key][current_body_subsection_key].append(p_block)
                else:
                    sections[current_section_key].append(p_block)
                is_heading_and_processed = True
            
            elif p_block["type"] == "heading_2" and current_section_key == "body":
                heading_text = p_block["plain_text"].lower()
                if "1분" in heading_text and "3분" in heading_text:
                    current_body_subsection_key = "1분_3분"
                elif "3분" in heading_text and "6분" in heading_text:
                    current_body_subsection_key = "3분_6분"
                elif "6분" in heading_text and "이후" in heading_text:
                    current_body_subsection_key = "6분_이후"
                else:
                    current_body_subsection_key = "other_h2_content" # For other H2s in body
                sections["body"][current_body_subsection_key].append(p_block)
                is_heading_and_processed = True

            if not is_heading_and_processed:
                if current_section_key == "body":
                    # Add block to the current body H2 subsection (or main_content if no H2 encountered yet)
                    sections[current_section_key][current_body_subsection_key].append(p_block)
                else:
                    # Add non-heading blocks to the current main section
                    sections[current_section_key].append(p_block)
            
        # Filter out empty sections from the final output for cleanliness, though initialized lists are fine
        # final_sections = {k: v for k, v in sections.items() if v} 
        # Keeping all sections, even if empty, for consistent structure.

        logger.info(f"Tool output: parse_notion_page_into_sections returned sections: {list(sections.keys())}")
        return sections

    except NotionAPIError as e:
        logger.error(f"Notion API error in parse_notion_page_into_sections for {page_id}: {str(e)}")
        raise Exception(f"Notion API error parsing page sections for {page_id}: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in parse_notion_page_into_sections for {page_id}: {str(e)}", exc_info=True)
        raise Exception(f"Unexpected error parsing page sections for {page_id}: {str(e)}")

# Export all tools
notion_tools = [
    get_page_tool,
    get_page_paragraph_text_blocks_tool,
    get_page_comment_content_blocks_tool,
    get_block_comments_tool,
    insert_block_comment_tool,
    get_page_title_tool,
    insert_page_comment_tool,
    update_page_properties_tool,
    get_page_blocks_tool,
    get_notion_page_blocks_tool,
    parse_notion_page_into_sections_tool,
] 