"""
Notion Agent using LangChain and LangGraph

This agent provides tools to interact with Notion API using LangChain and LangGraph.
"""

from typing import Dict, List, Optional, Type, Any, Annotated, Sequence, TypedDict
import os
import logging
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.tools import tools_by_name
from src.prompts import prompt_manager

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from phoenix.otel import register

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('notion_agent')

# Define the state type for our Page Evaluation Workflow
class PageEvaluationWorkflowState(TypedDict):
    """The state of the page evaluation workflow."""
    page_id: str
    title: Optional[str] # This is page title from page properties, not to be confused with section title
    status: Optional[str]
    parsed_sections: Optional[Dict[str, List[Dict]]] # Output of parse_notion_page_into_sections_tool
    title_evaluation_comment: Optional[str]
    thumbnail_evaluation_comment: Optional[str]
    intro_evaluation_comment: Optional[str]
    body_evaluation_comment: Optional[str]
    title_review_status: Optional[str] # Added for title review
    thumbnail_review_status: Optional[str] # Added for thumbnail review
    intro_review_status: Optional[str] # Added for intro review
    body_review_status: Optional[str] # Added for body review
    error_message: Optional[str]


# Define the Pydantic model for structured LLM output
class ReviewOutput(BaseModel):
    """Structured output for LLM review, containing comment and status."""
    comment: str = Field(description="The detailed review comment provided by the LLM.")
    status: str = Field(description="The review status, must be either 'approved' or 'change_requested'.")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4.1-nano").with_structured_output(ReviewOutput)

# Helper function to extract plain text from a list of block dictionaries
def _get_text_from_block_list(blocks: List[Dict]) -> str:
    all_text_parts = []
    if not blocks: # Handle empty list of blocks
        return "내용 없음"
        
    for block in blocks:
        # Ensure block is a dictionary and plain_text is a string before stripping
        if isinstance(block, dict) and isinstance(block.get("plain_text"), str):
            if block.get("type") not in ["quote", "heading_1", "heading_2"] and block.get("plain_text", "").strip():
                text = block.get("plain_text", "").strip()
                text = text.replace('"', '') # Corrected: Remove all double quotes
                all_text_parts.append(text)
        elif isinstance(block, dict) and block.get("type") in ["quote", "heading_1", "heading_2"]:
            pass # Skip these types intentionally
        elif isinstance(block, dict):
            logger.debug(f"Block has no plain_text or it's not a string: {block.get('id', 'N/A')}")
        else:
            logger.warning(f"Encountered non-dict block item in list: {type(block)}")

    if not all_text_parts:
        return "내용 없음" # If all blocks were filtered out or list was effectively empty
    return " ".join(all_text_parts)

def get_all_text_from_section(section_name: str, sections_data: Dict) -> str:
    """
    Extracts and concatenates plain text from all relevant blocks in a given section.
    Texts are joined by a single space.
    If the section_name is "body", it concatenates text from all paragraph blocks in all its sub-sections.
    Filters out "quote", "heading_1", "heading_2" types.
    """
    all_text_parts = []
    if section_name == "body":
        body_sub_sections_map = sections_data.get("body", {})
        if not isinstance(body_sub_sections_map, dict):
            logger.warning(f"Body section data is not a dict for page: {sections_data.get('page_id', 'Unknown')}. Type: {type(body_sub_sections_map)}. Returning '내용 없음'.")
            return "내용 없음"

        for _sub_key, blocks_in_sub_section_list in body_sub_sections_map.items():
            if not isinstance(blocks_in_sub_section_list, list):
                logger.warning(f"Blocks in body sub-section '{_sub_key}' is not a list for page: {sections_data.get('page_id', 'Unknown')}. Type: {type(blocks_in_sub_section_list)}. Skipping.")
                continue
            for block in blocks_in_sub_section_list:
                if isinstance(block, dict) and isinstance(block.get("plain_text"), str): # Ensure block is dict and plain_text is str
                    if block.get("type") not in ["quote", "heading_1", "heading_2"] and block.get("plain_text", "").strip():
                        text = block.get("plain_text", "").strip()
                        text = text.replace('"', '') # Corrected: Remove all double quotes
                        all_text_parts.append(text)
                elif isinstance(block, dict) and block.get("type") in ["quote", "heading_1", "heading_2"]:
                     pass # Skip these types intentionally
                elif isinstance(block, dict): # Has type but no plain_text or not string
                    logger.debug(f"Body section block {block.get('id', 'N/A')} has no plain_text or not string.")
                else: # Not a dict
                    logger.warning(f"Encountered non-dict block item in body sub-section '{_sub_key}': {type(block)}")

    else: # For sections other than "body"
        blocks_in_section_list = sections_data.get(section_name, [])
        if not isinstance(blocks_in_section_list, list):
            logger.warning(f"Blocks in section '{section_name}' is not a list for page: {sections_data.get('page_id', 'Unknown')}. Type: {type(blocks_in_section_list)}. Returning '내용 없음'.")
            return "내용 없음"
        for block in blocks_in_section_list:
            if isinstance(block, dict) and isinstance(block.get("plain_text"), str): # Ensure block is dict and plain_text is str
                if block.get("type") not in ["quote", "heading_1", "heading_2"] and block.get("plain_text", "").strip():
                    text = block.get("plain_text", "").strip()
                    text = text.replace('"', '') # Corrected: Remove all double quotes
                    all_text_parts.append(text)
            elif isinstance(block, dict) and block.get("type") in ["quote", "heading_1", "heading_2"]:
                pass # Skip these types intentionally
            elif isinstance(block, dict): # Has type but no plain_text or not string
                 logger.debug(f"Section '{section_name}' block {block.get('id', 'N/A')} has no plain_text or not string.")
            else: # Not a dict
                logger.warning(f"Encountered non-dict block item in section '{section_name}': {type(block)}")
    
    if not all_text_parts:
        return "내용 없음"
    return " ".join(all_text_parts)


# Define the node for getting page information
def get_page_info_node(state: PageEvaluationWorkflowState) -> Dict:
    """Fetches page properties (title, status) and parsed page content sections."""
    logger.info(f"Executing get_page_info_node for page_id: {state['page_id']}")
    page_id = state["page_id"]
    
    try:
        # 1. Get page properties
        page_props_tool = tools_by_name.get("get_notion_page")
        if not page_props_tool:
            logger.error("Tool 'get_notion_page' not found.")
            return {"error_message": "Tool 'get_notion_page' not found."}
        
        logger.info(f"Retrieving properties for page: {page_id}")
        page_properties = page_props_tool.invoke({"page_id": page_id})
        
        title_property = page_properties.get("properties", {}).get("title", {})
        title_text_array = title_property.get("title", [])
        actual_page_title = title_text_array[0].get("plain_text", "N/A") if title_text_array else "N/A"
        
        status_property = page_properties.get("properties", {}).get("status", {})
        status = status_property.get("status", {}).get("name", "N/A")
        logger.info(f"Page properties retrieved: Title='{actual_page_title}', Status='{status}'")

        # 2. Get parsed page content sections
        parse_sections_tool = tools_by_name.get("parse_notion_page_into_sections_tool") # User changed this in their diff
        if not parse_sections_tool:
            logger.error("Tool 'parse_notion_page_into_sections_tool' not found.")
            return {"error_message": "Tool 'parse_notion_page_into_sections_tool' not found."}

        logger.info(f"Retrieving and parsing sections for page: {page_id}")
        parsed_page_sections = parse_sections_tool.invoke({"page_id": page_id})
        
        logger.info(f"Retrieved {len(parsed_page_sections)} sections for page {page_id}: {list(parsed_page_sections.keys())}")
        
        return {
            "title": actual_page_title, # This is the page's actual title
            "status": status,
            "parsed_sections": parsed_page_sections,
            "error_message": None
        }
    except Exception as e:
        logger.error(f"Error in get_page_info_node for page {page_id}: {str(e)}", exc_info=True)
        return {
            "error_message": f"Failed to get page info: {str(e)}",
            "title": None,
            "status": None,
            "parsed_sections": None
            }

# Define the node for evaluating the title section
def evaluate_title_node(state: PageEvaluationWorkflowState) -> Dict:
    """Evaluates the title planning/description section of the page, using context from other sections."""
    page_id = state["page_id"]
    logger.info(f"Executing evaluate_title_node for page_id: {page_id}")

    if state.get("error_message"):
        logger.warning(f"Skipping title evaluation due to previous error: {state['error_message']}")
        return {"title_evaluation_comment": "Skipped due to previous error", "title_review_status": "change_requested"}

    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate title section.")
        return {"error_message": "Parsed sections not found for title evaluation.", "title_review_status": "change_requested"}

    title_section_blocks = parsed_sections.get("title", [])
    title_heading_block_id = None
    if title_section_blocks and isinstance(title_section_blocks[0], dict):
        title_heading_block_id = title_section_blocks[0].get("id")
    
    title_text = get_all_text_from_section("title", parsed_sections)
    
    if title_text == "내용 없음" or not title_text.strip():
        comment_text = "제목에 대한 설명/기획 내용(문단 블록)이 없습니다."
        logger.info(f"No descriptive content found in title section. Comment: '{comment_text}'")
        
        if title_heading_block_id:
            try:
                tools_by_name["insert_notion_block_comment"].invoke({"block_id": title_heading_block_id, "text": comment_text}) 
                logger.info(f"Block comment added to title heading block {title_heading_block_id} for missing content.")
            except Exception as e:
                logger.error(f"Error adding block comment for missing title description to {title_heading_block_id}: {str(e)}")
                # Fallback to page comment if block comment fails
                try:
                    page_fallback_comment = f"제목 섹션 내용 누락 확인: '{comment_text}' (블록 ID {title_heading_block_id} 코멘트 실패)"
                    tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_fallback_comment})
                except Exception as e_page:
                    logger.error(f"Error adding fallback page comment for missing title description: {str(e_page)}")
                return {"title_evaluation_comment": comment_text, "title_review_status": "change_requested", "error_message": f"Error adding comment for title: {str(e)}"}
        else:
            logger.warning("Title heading block ID not found. Adding page comment for missing title content.")
            try:
                page_comment_text = "제목 섹션의 제목 블록이 없거나, 기획 내용(문단 블록)이 없습니다."
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text})
            except Exception as e:
                logger.error(f"Error adding page comment for missing title section/content: {str(e)}")
                return {"title_evaluation_comment": page_comment_text, "title_review_status": "change_requested", "error_message": f"Error adding page comment for title: {str(e)}"}
        return {"title_evaluation_comment": comment_text, "title_review_status": "change_requested", "error_message": None}
    
    title_plan_to_evaluate = title_text
    if title_heading_block_id:
        logger.info(f"Title plan text for direct evaluation: '{title_plan_to_evaluate[:100]}...' from block {title_heading_block_id}")
    else:
        logger.info(f"Title plan text for direct evaluation: '{title_plan_to_evaluate[:100]}...' (No specific heading block ID found for title section)")

    thumbnail_context = get_all_text_from_section("thumbnail", parsed_sections)
    intro_context = get_all_text_from_section("intro", parsed_sections)
    body_context = get_all_text_from_section("body", parsed_sections)

    logger.info(f"Context for title evaluation: Thumbnail='{thumbnail_context[:50]}...', Intro='{intro_context[:50]}...', Body='{body_context[:50]}...'")
    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt")
        system_prompt_content = str(system_prompt_template.format()) if hasattr(system_prompt_template, 'format') else str(system_prompt_template)
        
        human_message_content = (
            f"평가할 제목 기획: \n\"{title_plan_to_evaluate}\"\n\n"
            f"참고 컨텍스트:\n"
            f"- 썸네일 기획: \"{thumbnail_context}\"\n"
            f"- 인트로 기획: \"{intro_context}\"\n"
            f"- 본문 기획: \"{body_context}\"\n\n"
            f"위 제목 기획과 참고 컨텍스트를 바탕으로, 이 제목이 유튜브 영상의 클릭율을 높이는 데 도움이 될지 종합적으로 평가하고, \n"
            f"평가 상태를 'approved' 또는 'change_requested'로 지정해주세요."
        )

        messages = [
            SystemMessage(content=system_prompt_content.messages[0]["content"] if hasattr(system_prompt_content, 'messages') and system_prompt_content.messages else system_prompt_content), # Ensure content access is safe
            HumanMessage(content=human_message_content),
        ]
        
        structured_response: ReviewOutput = llm.invoke(messages)
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for title. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Title plan evaluation by LLM: Comment='{evaluation_comment}', Status='{review_status}'")

        if title_heading_block_id:
            tools_by_name["insert_notion_block_comment"].invoke({"block_id": title_heading_block_id, "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
            logger.info(f"Block comment added to title plan block {title_heading_block_id}")
        else:
            logger.warning(f"No title heading block ID found. Adding page comment for title evaluation. Page ID: {page_id}")
            page_comment_text = f"제목 평가: {evaluation_comment} (상태: {review_status}) - 세부 블록 코멘트 불가 (제목 블록 ID 누락)"
            try:
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text })
                logger.info("Page comment added for title evaluation as fallback.")
            except Exception as e_page:
                logger.error(f"Error adding fallback page comment for title evaluation: {str(e_page)}")
        
        return {"title_evaluation_comment": evaluation_comment, "title_review_status": review_status, "error_message": None}
    except Exception as e:
        logger.error(f"Error during title plan evaluation or commenting: {str(e)}", exc_info=True)
        return {
            "error_message": f"Error during title plan evaluation: {str(e)}", 
            "title_evaluation_comment": "Error during evaluation.", 
            "title_review_status": "change_requested"
            }

# Define the node for evaluating the thumbnail section
def evaluate_thumbnail_node(state: PageEvaluationWorkflowState) -> Dict:
    """Evaluates the thumbnail planning/description section of the page."""
    page_id = state["page_id"]
    logger.info(f"Executing evaluate_thumbnail_node for page_id: {page_id}")

    if state.get("error_message"):
        logger.warning(f"Skipping thumbnail evaluation due to previous error: {state['error_message']}")
        return {"thumbnail_evaluation_comment": "Skipped due to previous error", "thumbnail_review_status": "change_requested"}
    
    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate thumbnail section.")
        return {"error_message": "Parsed sections not found for thumbnail evaluation.", "thumbnail_review_status": "change_requested"}

    thumbnail_section_blocks = parsed_sections.get("thumbnail", [])
    thumbnail_heading_block_id = None
    if thumbnail_section_blocks and isinstance(thumbnail_section_blocks[0], dict):
        thumbnail_heading_block_id = thumbnail_section_blocks[0].get("id")
        
    thumbnail_text = get_all_text_from_section("thumbnail", parsed_sections)

    if thumbnail_text == "내용 없음" or not thumbnail_text.strip():
        comment_text = "썸네일 기획 내용(문단 블록)이 없습니다."
        logger.info(f"No descriptive content found in thumbnail section. Comment: '{comment_text}'")
        if thumbnail_heading_block_id:
            try:
                tools_by_name["insert_notion_block_comment"].invoke({"block_id": thumbnail_heading_block_id, "text": comment_text})
                logger.info(f"Block comment added to thumbnail heading block {thumbnail_heading_block_id} for missing content.")
            except Exception as e:
                logger.error(f"Error adding block comment for missing thumbnail description to {thumbnail_heading_block_id}: {str(e)}")
                try:
                    page_fallback_comment = f"썸네일 섹션 내용 누락 확인: '{comment_text}' (블록 ID {thumbnail_heading_block_id} 코멘트 실패)"
                    tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_fallback_comment})
                except Exception as e_page:
                    logger.error(f"Error adding fallback page comment for missing thumbnail description: {str(e_page)}")
                return {"thumbnail_evaluation_comment": comment_text, "thumbnail_review_status": "change_requested", "error_message": f"Error adding comment for thumbnail: {str(e)}"}
        else:
            logger.warning("Thumbnail heading block ID not found. Adding page comment for missing thumbnail content.")
            try:
                page_comment_text = "썸네일 섹션의 제목 블록이 없거나, 기획 내용(문단 블록)이 없습니다."
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text})
            except Exception as e:
                logger.error(f"Error adding page comment for missing thumbnail section/content: {str(e)}")
                return {"thumbnail_evaluation_comment": page_comment_text, "thumbnail_review_status": "change_requested", "error_message": f"Error adding page comment for thumbnail: {str(e)}"}
        return {"thumbnail_evaluation_comment": comment_text, "thumbnail_review_status": "change_requested", "error_message": None}
    
    logger.info(f"Thumbnail plan text found for evaluation: '{thumbnail_text[:100]}...'")
    
    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt") # Assuming same prompt is okay for now
        system_prompt_content = str(system_prompt_template.format()) if hasattr(system_prompt_template, 'format') else str(system_prompt_template)

        human_message_content = (
            f"평가할 썸네일 기획: \n\"{thumbnail_text}\"\n\n"
            f"이 썸네일 기획이 유튜브 영상의 클릭율을 높이는 데 도움이 될지 종합적으로 평가하고, \n"
            f"평가 상태를 'approved' 또는 'change_requested'로 지정해주세요."
        )

        messages = [
            SystemMessage(content=system_prompt_content.messages[0]["content"] if hasattr(system_prompt_content, 'messages') and system_prompt_content.messages else system_prompt_content),
            HumanMessage(content=human_message_content),
        ]
        
        structured_response: ReviewOutput = llm.invoke(messages)
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for thumbnail. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Thumbnail plan evaluation by LLM: Comment='{evaluation_comment}', Status='{review_status}'")

        if thumbnail_heading_block_id:
            tools_by_name["insert_notion_block_comment"].invoke({"block_id": thumbnail_heading_block_id, "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
            logger.info(f"Block comment added to thumbnail plan block {thumbnail_heading_block_id}")
        else:
            logger.warning(f"No thumbnail heading block ID found. Adding page comment for thumbnail evaluation. Page ID: {page_id}")
            page_comment_text = f"썸네일 평가: {evaluation_comment} (상태: {review_status}) - 세부 블록 코멘트 불가 (썸네일 블록 ID 누락)"
            try:
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text })
                logger.info("Page comment added for thumbnail evaluation as fallback.")
            except Exception as e_page:
                logger.error(f"Error adding fallback page comment for thumbnail evaluation: {str(e_page)}")

        return {"thumbnail_evaluation_comment": evaluation_comment, "thumbnail_review_status": review_status, "error_message": None}
    except Exception as e:
        logger.error(f"Error during thumbnail plan evaluation or commenting: {str(e)}", exc_info=True)
        return {
            "error_message": f"Error during thumbnail plan evaluation: {str(e)}", 
            "thumbnail_evaluation_comment": "Error during evaluation.",
            "thumbnail_review_status": "change_requested"
            }


# Define the node for evaluating the introduction
def evaluate_intro_node(state: PageEvaluationWorkflowState) -> Dict:
    """Evaluates the intro of the page. 
    Comments on heading block if available, otherwise on page."""
    logger.info(f"Executing evaluate_intro_node for page_id: {state['page_id']}")
    page_id = state["page_id"]

    if state.get("error_message"):
        logger.warning(f"Skipping intro evaluation due to previous error: {state['error_message']}")
        return {"intro_evaluation_comment": "Skipped due to previous error", "intro_review_status": "change_requested"}
    
    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate intro section.")
        return {"error_message": "Parsed sections not found for intro evaluation.", "intro_review_status": "change_requested"}

    intro_section_blocks = parsed_sections.get("intro", [])
    intro_heading_block_id = None
    if intro_section_blocks and isinstance(intro_section_blocks[0], dict):
        intro_heading_block_id = intro_section_blocks[0].get("id")
        
    intro_plan_text = get_all_text_from_section("intro", parsed_sections)

    if intro_plan_text == "내용 없음" or not intro_plan_text.strip():
        comment_text_block = "인트로 기획 내용(문단 블록)이 없습니다."
        logger.info(f"No descriptive content found in intro section.")
        
        if intro_heading_block_id:
            logger.info(f"Commenting on intro heading block {intro_heading_block_id} for missing content.")
            try:
                tools_by_name["insert_notion_block_comment"].invoke({"block_id": intro_heading_block_id, "text": comment_text_block})
            except Exception as e:
                logger.error(f"Error adding block comment for missing intro content to {intro_heading_block_id}: {str(e)}")
                try:
                    page_fallback_comment = f"인트로 섹션 내용 누락 확인: '{comment_text_block}' (블록 ID {intro_heading_block_id} 코멘트 실패)"
                    tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_fallback_comment})
                except Exception as e_page:
                    logger.error(f"Error adding fallback page comment for missing intro content: {str(e_page)}")
                return {"intro_evaluation_comment": comment_text_block, "intro_review_status": "change_requested", "error_message": f"Error adding comment for intro: {str(e)}"}
            return {"intro_evaluation_comment": comment_text_block, "intro_review_status": "change_requested", "error_message": None}
        else:
            # Fallback to more detailed page comment if heading block itself is missing
            intro_heading_exists_in_blocks = any(
                block.get("type") == "heading_1" and 
                isinstance(block.get("plain_text"), str) and
                ("인트로" in block["plain_text"].lower() or "초반 30초" in block["plain_text"].lower()) 
                for block in intro_section_blocks # Check within all blocks of intro section
            )
            if intro_heading_exists_in_blocks: # Heading text found, but maybe not as first block or no ID
                 page_comment_text = "인트로 섹션은 있으나, 세부 기획 내용(문단 블록)이 없습니다. (제목 블록 ID를 특정할 수 없어 페이지에 코멘트합니다)"
            else:
                 page_comment_text = "인트로 섹션 자체가 없거나, 인식할 수 있는 인트로 관련 제목(예: '인트로' 또는 '초반 30초' H1)을 찾지 못했습니다."
            logger.warning(f"Intro heading block ID not found or section malformed. Adding page comment: '{page_comment_text}'")
            try:
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text})
            except Exception as e:
                logger.error(f"Error adding page comment for missing intro section/content: {str(e)}")
                return {"intro_evaluation_comment": page_comment_text, "intro_review_status": "change_requested", "error_message": f"Error adding page comment for intro: {str(e)}"}
            return {"intro_evaluation_comment": page_comment_text, "intro_review_status": "change_requested", "error_message": None}

    if intro_heading_block_id:
        logger.info(f"Intro plan text found for evaluation: '{intro_plan_text[:100]}...' from block {intro_heading_block_id}")
    else:
        logger.info(f"Intro plan text found for evaluation: '{intro_plan_text[:100]}...' (No specific heading block ID for intro section)")
        
    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-intro-evaluation-prompt")
        system_prompt_content = str(system_prompt_template.format()) if hasattr(system_prompt_template, 'format') else str(system_prompt_template)

        human_message_content = (
            f"평가할 인트로 기획: \n\"{intro_plan_text}\"\n\n"
            f"이 인트로 기획이 유튜브 영상의 초반 이탈율을 줄이는 데 도움이 될지 평가해주세요. \n"
            f"평가 상태를 'approved' 또는 'change_requested'로 지정해주세요."
        )

        messages = [
            SystemMessage(content=system_prompt_content.messages[0]["content"] if hasattr(system_prompt_content, 'messages') and system_prompt_content.messages else system_prompt_content),
            HumanMessage(content=human_message_content),
        ]
        
        structured_response: ReviewOutput = llm.invoke(messages)
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for intro. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Intro plan evaluation by LLM: Comment='{evaluation_comment}', Status='{review_status}'")

        if intro_heading_block_id: 
            tools_by_name["insert_notion_block_comment"].invoke({"block_id": intro_heading_block_id, "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
            logger.info(f"Block comment added to intro plan block {intro_heading_block_id}")
        else:
            logger.warning(f"No block ID for intro section, adding page comment for evaluation. Page ID: {page_id}")
            page_comment_text = f"인트로 평가: {evaluation_comment} (상태: {review_status}) - 세부 블록 코멘트 불가 (인트로 제목 블록 ID 누락)"
            try:
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text })
                logger.info("Page comment added for intro evaluation as fallback.")
            except Exception as e_page:
                logger.error(f"Error adding fallback page comment for intro evaluation: {str(e_page)}")


        return {"intro_evaluation_comment": evaluation_comment, "intro_review_status": review_status, "error_message": None}
    except Exception as e:
        logger.error(f"Error during intro plan evaluation or commenting: {str(e)}", exc_info=True)
        return {
            "error_message": f"Error during intro plan evaluation: {str(e)}", 
            "intro_evaluation_comment": "Error during evaluation.",
            "intro_review_status": "change_requested"
            }

# Define the node for evaluating the body
def evaluate_body_node(state: PageEvaluationWorkflowState) -> Dict:
    """Evaluates the body planning/description section of the page.
    Comments on H1 '본문' block if available, otherwise on page."""
    logger.info(f"Executing evaluate_body_node for page_id: {state['page_id']}")
    page_id = state["page_id"]

    if state.get("error_message"):
        logger.warning(f"Skipping body evaluation due to previous error: {state['error_message']}")
        return {"body_evaluation_comment": "Skipped due to previous error", "body_review_status": "change_requested"}
    
    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate body section.")
        return {"error_message": "Parsed sections not found for body evaluation.", "body_review_status": "change_requested"}

    # Overall body text for initial empty check and summary
    body_plan_text = get_all_text_from_section("body", parsed_sections)

    body_data = parsed_sections.get("body", {}) # This should be Dict[str, List[Dict]]
    if not isinstance(body_data, dict): # Safeguard if parsed_sections["body"] is not a dict
        logger.error(f"Body data in parsed_sections is not a dictionary for page {page_id}. Type: {type(body_data)}")
        # Treat as if body is empty for commenting purposes, but log error
        body_data = {} # Ensure it's a dict for subsequent .get calls to not fail
        body_plan_text = "내용 없음" # Force this condition for commenting logic

    body_main_content_blocks = body_data.get("main_content", [])
    
    body_h1_block_id = None
    # Try to find H1 "본문" block ID from main_content
    if body_main_content_blocks and isinstance(body_main_content_blocks[0], dict) and body_main_content_blocks[0].get("type") == "heading_1":
        # Check if the first H1 in main_content is indeed "본문"
        h1_text = body_main_content_blocks[0].get("plain_text", "").lower()
        if "본문" in h1_text:
            body_h1_block_id = body_main_content_blocks[0].get("id")
        else: # First H1 is not "본문", search other H1s in main_content
            for block in body_main_content_blocks:
                if isinstance(block, dict) and block.get("type") == "heading_1" and "본문" in block.get("plain_text", "").lower():
                    body_h1_block_id = block.get("id")
                    break
            if not body_h1_block_id: # If no "본문" H1, but there's an H1, use the first one's ID.
                 body_h1_block_id = body_main_content_blocks[0].get("id")
                 logger.info(f"Using first H1 block {body_h1_block_id} in body main_content as no specific '본문' H1 found for commenting.")

    elif body_main_content_blocks and isinstance(body_main_content_blocks[0], dict): # First block is not H1
         logger.info(f"First block in body main_content is {body_main_content_blocks[0].get('type')}, not heading_1. Searching for any '본문' H1 for commenting.")
         for block in body_main_content_blocks:
            if isinstance(block, dict) and block.get("type") == "heading_1" and "본문" in block.get("plain_text", "").lower():
                body_h1_block_id = block.get("id")
                break


    if body_plan_text == "내용 없음" or not body_plan_text.strip():
        comment_text_block = "본문 기획 내용(문단 블록)이 없습니다."
        logger.info(f"No descriptive content found in body section for page {page_id}.")

        if body_h1_block_id:
            logger.info(f"Commenting on body H1 block {body_h1_block_id} for missing content.")
            try:
                tools_by_name["insert_notion_block_comment"].invoke({"block_id": body_h1_block_id, "text": comment_text_block})
            except Exception as e:
                logger.error(f"Error adding block comment for missing body content to {body_h1_block_id}: {str(e)}")
                try:
                    page_fallback_comment = f"본문 섹션 내용 누락 확인: '{comment_text_block}' (블록 ID {body_h1_block_id} 코멘트 실패)"
                    tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_fallback_comment})
                except Exception as e_page:
                    logger.error(f"Error adding fallback page comment for missing body content: {str(e_page)}")
                return {"body_evaluation_comment": comment_text_block, "body_review_status": "change_requested", "error_message": f"Error adding comment for body: {str(e)}"}
            return {"body_evaluation_comment": comment_text_block, "body_review_status": "change_requested", "error_message": None}
        else:
            body_h1_exists_in_main_content = any(
                isinstance(block, dict) and block.get("type") == "heading_1" and 
                isinstance(block.get("plain_text"), str) and "본문" in block.get("plain_text", "").lower()
                for block in body_main_content_blocks
            )
            if body_h1_exists_in_main_content:
                 page_comment_text = "본문 섹션은 있으나 (H1 '본문' 확인됨), 세부 기획 내용(문단 블록)이 없습니다. (H1 블록 ID를 특정할 수 없어 페이지에 코멘트합니다)"
            else:
                 page_comment_text = "본문 섹션 자체가 없거나, 인식할 수 있는 본문 관련 H1 제목('본문')을 찾지 못했습니다."
            logger.warning(f"Body H1 block ID not found or section malformed for page {page_id}. Adding page comment: '{page_comment_text}'")
            try:
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text})
            except Exception as e:
                logger.error(f"Error adding page comment for missing body section/content: {str(e)}")
                return {"body_evaluation_comment": page_comment_text, "body_review_status": "change_requested", "error_message": f"Error adding page comment for body: {str(e)}"}
            return {"body_evaluation_comment": page_comment_text, "body_review_status": "change_requested", "error_message": None}

    # Extract text from specific body sub-sections using the helper
    main_content_text = _get_text_from_block_list(body_data.get("main_content", []))
    one_to_three_text = _get_text_from_block_list(body_data.get("1분_3분", []))
    three_to_six_text = _get_text_from_block_list(body_data.get("3분_6분", []))
    six_plus_text = _get_text_from_block_list(body_data.get("6분_이후", []))
    other_h2_text = _get_text_from_block_list(body_data.get("other_h2_content", []))

    log_message_parts = [f"Body plan text for page {page_id} (overall): '{body_plan_text[:100]}...'"]
    if body_h1_block_id:
        log_message_parts.append(f"Commenting on H1 block {body_h1_block_id}.")
    else:
        log_message_parts.append("No specific H1 '본문' block ID found for direct commenting.")
    logger.info(" ".join(log_message_parts))
    logger.info(f"Sub-section texts: Main='{main_content_text[:30]}...', 1-3='{one_to_three_text[:30]}...', 3-6='{three_to_six_text[:30]}...', 6+='{six_plus_text[:30]}...', OtherH2='{other_h2_text[:30]}...'")

    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt") # Assuming same prompt for structure
        system_prompt_content = str(system_prompt_template.format()) if hasattr(system_prompt_template, 'format') else str(system_prompt_template)
        
        detailed_body_segments = []
        if main_content_text != "내용 없음" and main_content_text.strip():
            detailed_body_segments.append(f"- 주요 내용 (H1 '본문' 제목 바로 아래 단락들): \"{main_content_text}\"")
        if one_to_three_text != "내용 없음" and one_to_three_text.strip():
            detailed_body_segments.append(f"- '1분-3분' 구간 추정 내용: \"{one_to_three_text}\"")
        if three_to_six_text != "내용 없음" and three_to_six_text.strip():
            detailed_body_segments.append(f"- '3분-6분' 구간 추정 내용: \"{three_to_six_text}\"")
        if six_plus_text != "내용 없음" and six_plus_text.strip():
            detailed_body_segments.append(f"- '6분 이후' 구간 추정 내용: \"{six_plus_text}\"")
        if other_h2_text != "내용 없음" and other_h2_text.strip():
            detailed_body_segments.append(f"- 기타 H2 제목 하위 내용: \"{other_h2_text}\"")
        
        segments_prompt_text = "\n".join(detailed_body_segments) if detailed_body_segments else "세부적으로 구분된 본문 내용이 없습니다."

        human_message_content = (
            f"평가할 본문 기획입니다. 먼저 전체적으로 파악된 본문 내용은 다음과 같습니다:\\n"
            f"\"{body_plan_text}\"\\n\\n"
            f"이 본문은 다음과 같은 세부적인 구성으로 나뉘어 있을 수 있습니다 (각 구간의 내용이 없다면 '내용 없음' 또는 생략됩니다):\\n"
            f"{segments_prompt_text}\\n\\n"
            f"이 본문 기획과 위의 세부 구성안이 유튜브 영상의 평균 시청 지속 시간을 높이는 데 도움이 될지, "
            f"그리고 각 세부 내용이 전체적인 이야기 흐름과 시청자 몰입에 어떻게 기여하는지 종합적으로 평가해주세요.\\n"
            f"평가 상태를 'approved' 또는 'change_requested'로 지정해주세요."
        )
        
        messages = [
            SystemMessage(content=system_prompt_content.messages[0]["content"] if hasattr(system_prompt_content, 'messages') and system_prompt_content.messages else system_prompt_content),
            HumanMessage(content=human_message_content),
        ]
        
        structured_response: ReviewOutput = llm.invoke(messages)
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for body. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Body plan evaluation by LLM for page {page_id}: Comment='{evaluation_comment}', Status='{review_status}'")

        if body_h1_block_id: 
            tools_by_name["insert_notion_block_comment"].invoke({"block_id": body_h1_block_id, "text": f"본문 전체 평가: {evaluation_comment}\n상태: {review_status}"})
            logger.info(f"Block comment added to body H1 block {body_h1_block_id}")
        else:
            logger.warning(f"No H1 '본문' block ID found for body section on page {page_id}. Adding page comment for evaluation.")
            page_comment_text = f"본문 전체 평가: {evaluation_comment} (상태: {review_status}) - 세부 블록 코멘트 불가 (본문 H1 블록 ID 누락)"
            try:
                tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": page_comment_text })
                logger.info("Page comment added for body evaluation as fallback.")
            except Exception as e_page:
                logger.error(f"Error adding fallback page comment for body evaluation: {str(e_page)}")

        return {"body_evaluation_comment": evaluation_comment, "body_review_status": review_status, "error_message": None}
    except Exception as e:
        logger.error(f"Error during body plan evaluation or commenting for page {page_id}: {str(e)}", exc_info=True)
        return {
            "error_message": f"Error during body plan evaluation: {str(e)}", 
            "body_evaluation_comment": "Error during evaluation.",
            "body_review_status": "change_requested"
            }


def create_page_evaluator_agent():
    """Create and return a LangGraph-based agent for page evaluation."""
    logger.info("Creating page evaluation agent")
    
    workflow = StateGraph(PageEvaluationWorkflowState)
    
    # Add nodes
    workflow.add_node("get_page_info", get_page_info_node)
    workflow.add_node("evaluate_title", evaluate_title_node)
    workflow.add_node("evaluate_thumbnail", evaluate_thumbnail_node)
    workflow.add_node("evaluate_intro", evaluate_intro_node)
    workflow.add_node("evaluate_body", evaluate_body_node)
    
    # Set the entry point
    workflow.set_entry_point("get_page_info")
    
    # Define transitions
    workflow.add_edge("get_page_info", "evaluate_title")
    workflow.add_edge("evaluate_title", "evaluate_thumbnail")
    workflow.add_edge("evaluate_thumbnail", "evaluate_intro")
    workflow.add_edge("evaluate_intro", "evaluate_body")
    workflow.add_edge("evaluate_body", END)
    
    # Compile the graph
    checkpointer = InMemorySaver() # Optional: for state persistence if needed across runs
    agent = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Page evaluation agent created successfully")
    return agent

def run_page_evaluator_agent(page_id: str):
    """Run the page evaluation agent with a specific page_id."""
    logger.info(f"Running page evaluation agent for page_id: {page_id}")
    
    agent = create_page_evaluator_agent()
    
    initial_state = {"page_id": page_id} 
    config = {"configurable": {"thread_id": f"page_eval_{page_id}"}} # Unique thread_id for each run

    final_state = None
    try:
        # Directly invoke the agent to get the final state
        final_state = agent.invoke(initial_state, config=config)
        logger.info(f"Agent invoked successfully. Final state: {final_state}")

    except Exception as e:
        logger.error(f"Error running page evaluation agent for page_id {page_id}: {str(e)}", exc_info=True)
        # Construct a consistent error state if invoke fails
        final_state = {
            "page_id": page_id,
            "title": None,
            "status": None,
            "parsed_sections": None,
            "title_evaluation_comment": None,
            "thumbnail_evaluation_comment": None,
            "intro_evaluation_comment": None,
            "body_evaluation_comment": None,
            "title_review_status": None,
            "thumbnail_review_status": None,
            "intro_review_status": None,
            "body_review_status": None,
            "error_message": f"Agent execution failed: {str(e)}"
        }

    logger.info(f"Page evaluation agent execution completed for page_id: {page_id}")
    
    # Ensure final_state is a dictionary before constructing the summary
    if not isinstance(final_state, dict):
        logger.error(f"Critical error: final_state is not a dictionary. Value: {final_state}")
        final_state = {
            "page_id": page_id,
            "error_message": "Critical agent error: Final state was not a dictionary."
        }

    # Construct a summary response using final_state
    response_summary = {
        "page_id": page_id,
        "title": final_state.get("title", "N/A"),
        "status": final_state.get("status", "N/A"),
        "title_evaluation_comment": final_state.get("title_evaluation_comment", "N/A"),
        "thumbnail_evaluation_comment": final_state.get("thumbnail_evaluation_comment", "N/A"),
        "intro_evaluation_comment": final_state.get("intro_evaluation_comment", "N/A"),
        "body_evaluation_comment": final_state.get("body_evaluation_comment", "N/A"),
        "title_review_status": final_state.get("title_review_status"),
        "thumbnail_review_status": final_state.get("thumbnail_review_status"),
        "intro_review_status": final_state.get("intro_review_status"),
        "body_review_status": final_state.get("body_review_status"),
        "error_message": final_state.get("error_message")
    }
    return response_summary

# Create the Notion agent instance (for API use)
# notion_agent = create_notion_agent() # Old agent
page_evaluator_agent = create_page_evaluator_agent() # New agent instance for potential API use

# Example usage
if __name__ == "__main__":
    print("Notion Page Evaluator Agent 테스트 시작")
    print("=" * 50)
    
    # Test with a specific page ID
    test_page_id = "1f7ff0df284780bf9c00c14a6dc6af9f" # Replace with a valid page ID for testing
    
    print(f"\n테스트 페이지 ID: {test_page_id}")
    print("-" * 50)
    try:
        response = run_page_evaluator_agent(test_page_id)
        print(f"최종 응답 요약:")
        for key, value in response.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
    print("-" * 50)
    
    print("\nNotion Page Evaluator Agent 테스트 완료")
    print("=" * 50)
