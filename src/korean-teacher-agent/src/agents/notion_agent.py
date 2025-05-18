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

from src.tools import all_tools, tools_by_name
from src.prompts import prompt_manager # Removed as it's no longer used

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

# Initialize OpenTelemetry with Phoenix
# Phoenix server 도메인으로 환경 변수 설정

phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
phoenix_secret = os.getenv("PHOENIX_API_KEY", "")


logger.info(f"Phoenix endpoint: {phoenix_endpoint}")
logger.info(f"Phoenix API key configured: {bool(phoenix_secret)}")

 
# register 함수로 설정 초기화 (API 키가 있으면 자동으로 Authorization 헤더 추가)
tracer_provider = register(
    project_name="notion-agent",
    protocol="grpc",
    endpoint=phoenix_endpoint,
    headers={"Authorization": f"Bearer {phoenix_secret}"} if phoenix_secret else {},
    auto_instrument=True
)   

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

def get_first_text_from_section(section_name: str, sections_data: Dict) -> str:
    blocks_in_section = sections_data.get(section_name, [])
    for block in blocks_in_section:
        if block.get("type") == "paragraph" and block.get("plain_text", "").strip():
            return block.get("plain_text", "").strip()
    return "내용 없음"


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
        return {"title_evaluation_comment": "Skipped due to previous error"}

    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate title section.")
        return {"error_message": "Parsed sections not found for title evaluation."}

    title_section_blocks = parsed_sections.get("title", [])
    # User changed to look for paragraph blocks for the title description
    title_paragraph_blocks = [block for block in title_section_blocks if block.get("type") == "paragraph" and block.get("plain_text", "").strip()]
    
    if not title_paragraph_blocks:
        comment_text = "제목에 대한 설명/기획 내용(문단 블록)이 없습니다."
        logger.info(f"No descriptive paragraph found in title section. Page comment: '{comment_text}'")
        try:
            # Ensure the tool name and parameters match its definition in notion_tools.py
            tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": comment_text}) 
            return {"title_evaluation_comment": comment_text, "title_review_status": "change_requested", "error_message": None}
        except Exception as e:
            logger.error(f"Error adding page comment for missing title description: {str(e)}")
            return {"error_message": f"Error adding page comment for title: {str(e)}", "title_review_status": "change_requested"}
    
    title_plan_to_evaluate = title_paragraph_blocks[0].get("plain_text", "")
    title_plan_block_id = title_paragraph_blocks[0].get("id")
    logger.info(f"Title plan text for direct evaluation: '{title_plan_to_evaluate[:100]}...' from block {title_plan_block_id}")
    # Gather context from other sections
    thumbnail_context = get_first_text_from_section("thumbnail", parsed_sections)
    intro_context = get_first_text_from_section("intro", parsed_sections)
    body_context = get_first_text_from_section("body", parsed_sections)

    logger.info(f"Context for title evaluation: Thumbnail='{thumbnail_context[:50]}...', Intro='{intro_context[:50]}...', Body='{body_context[:50]}...'")
    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt")
        try:
            system_prompt_content = system_prompt_template.format() 
        except AttributeError:
            system_prompt_content = str(system_prompt_template) 
        
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
            SystemMessage(content=system_prompt_content.messages[0]["content"]),
            HumanMessage(content=human_message_content),
        ]
        
        # LLM now returns a ReviewOutput object directly
        structured_response: ReviewOutput = llm.invoke(messages)
        
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for title. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Title plan evaluation by LLM: Comment='{evaluation_comment}', Status='{review_status}'")

        tools_by_name["insert_notion_block_comment"].invoke({"block_id": title_plan_block_id, "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
        logger.info(f"Block comment added to title plan block {title_plan_block_id}")
        return {"title_evaluation_comment": evaluation_comment, "title_review_status": review_status, "error_message": None}
    except Exception as e:
        logger.error(f"Error during title plan evaluation or commenting: {str(e)}", exc_info=True) # Added exc_info for better debugging
        # If LLM fails or structured output parsing fails within LangChain, it might raise an error caught here
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
        return {"thumbnail_evaluation_comment": "Skipped due to previous error"}
    
    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate thumbnail section.")
        return {"error_message": "Parsed sections not found for thumbnail evaluation."}

    thumbnail_section_blocks = parsed_sections.get("thumbnail", [])
    thumbnail_text = get_first_text_from_section("thumbnail", parsed_sections)

    if thumbnail_text == "내용 없음" or not thumbnail_text.strip():
        comment_text = "썸네일 기획 내용이 없습니다."
        logger.info(f"No descriptive quote found in thumbnail section. Page comment: '{comment_text}'")
        try:
            # Ensure the tool name and parameters match its definition
            tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": comment_text}) 
            return {"thumbnail_evaluation_comment": comment_text, "thumbnail_review_status": "change_requested", "error_message": None}
        except Exception as e:
            logger.error(f"Error adding page comment for missing thumbnail description: {str(e)}")
            return {"error_message": f"Error adding page comment for thumbnail: {str(e)}", "thumbnail_review_status": "change_requested"}
    
    logger.info(f"Thumbnail plan text found for evaluation: '{thumbnail_text[:100]}...'")
    
    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt")
        try:
            system_prompt_content = system_prompt_template.format()
        except AttributeError:
            system_prompt_content = str(system_prompt_template)

        human_message_content = (
            f"평가할 썸네일 기획: \n\"{thumbnail_text}\"\n\n"
            f"이 썸네일 기획이 유튜브 영상의 클릭율을 높이는 데 도움이 될지 종합적으로 평가하고, \n"
            f"평가 상태를 'approved' 또는 'change_requested'로 지정해주세요."
        )

        messages = [
            SystemMessage(content=system_prompt_content.messages[0]["content"] if hasattr(system_prompt_content, 'messages') and system_prompt_content.messages else system_prompt_content),
            HumanMessage(content=human_message_content),
        ]
        
        # LLM now returns a ReviewOutput object directly
        structured_response: ReviewOutput = llm.invoke(messages)
        
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for thumbnail. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Thumbnail plan evaluation by LLM: Comment='{evaluation_comment}', Status='{review_status}'")

        tools_by_name["insert_notion_block_comment"].invoke({"block_id": thumbnail_section_blocks[0].get("id"), "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
        logger.info(f"Block comment added to thumbnail plan block {thumbnail_section_blocks[0].get('id')}")
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
    If no intro, comments on the page. 
    If intro exists, evaluates and comments on the first intro block."""
    logger.info(f"Executing evaluate_intro_node for page_id: {state['page_id']}")
    page_id = state["page_id"]

    if state.get("error_message"):
        logger.warning(f"Skipping intro evaluation due to previous error: {state['error_message']}")
        return {"intro_evaluation_comment": "Skipped due to previous error"}
    
    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate intro section.")
        return {"error_message": "Parsed sections not found for intro evaluation."}

    intro_section_blocks = parsed_sections.get("intro", [])
    intro_plan_text = get_first_text_from_section("intro", parsed_sections)

    if intro_plan_text == "내용 없음" or not intro_plan_text.strip():
        # Check if there was at least a heading for the intro section
        intro_heading_exists = any(block.get("type") == "heading_1" and ("인트로" in block.get("plain_text", "").lower() or "초반 30초" in block.get("plain_text", "").lower()) for block in intro_section_blocks)
        
        if intro_heading_exists:
            comment_text = "인트로 섹션은 있으나, 세부 기획 내용(문단 블록)이 없습니다."
        else:
            comment_text = "인트로 섹션 자체가 없거나, 인식할 수 있는 인트로 관련 내용(예: '인트로' 또는 '초반 30초' 제목)을 찾지 못했습니다."
        
        logger.info(f"No descriptive paragraph found in intro section. Page comment: '{comment_text}'")
        try:
            tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": comment_text})
            return {"intro_evaluation_comment": comment_text, "intro_review_status": "change_requested", "error_message": None}
        except Exception as e:
            logger.error(f"Error adding page comment for missing intro description: {str(e)}")
            return {"error_message": f"Error adding page comment for intro: {str(e)}", "intro_review_status": "change_requested"}

    # Get block_id from the first block in the section if blocks exist
    intro_plan_block_id = None
    if intro_section_blocks:
        intro_plan_block_id = intro_section_blocks[0].get("id")
        logger.info(f"Intro plan text found for evaluation: '{intro_plan_text[:100]}...' from section. Comment will be on block_id: {intro_plan_block_id}")
    else: # Should not happen if intro_plan_text was found, but as a safeguard
        logger.warning(f"Intro plan text found ('{intro_plan_text[:100]}...') but no blocks in intro_section_blocks. Cannot comment on block.")
        # Decide if we should still proceed with evaluation but not comment, or return error
        # For now, proceed but commenting will fail if block_id is None

    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt")
        try:
            system_prompt_content = system_prompt_template.format()
        except AttributeError:
            system_prompt_content = str(system_prompt_template)

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

        if intro_plan_block_id: # Only comment if we have a block_id
            tools_by_name["insert_notion_block_comment"].invoke({"block_id": intro_plan_block_id, "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
            logger.info(f"Block comment added to intro plan block {intro_plan_block_id}")
        else:
            logger.warning(f"No block ID for intro section, skipping block comment. Page ID: {page_id}")

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
    """Evaluates the body planning/description section of the page."""
    logger.info(f"Executing evaluate_body_node for page_id: {state['page_id']}")
    page_id = state["page_id"]

    if state.get("error_message"):
        logger.warning(f"Skipping body evaluation due to previous error: {state['error_message']}")
        return {"body_evaluation_comment": "Skipped due to previous error"}
    
    parsed_sections = state.get("parsed_sections")
    if not parsed_sections:
        logger.warning("No parsed sections found. Cannot evaluate body section.")
        return {"error_message": "Parsed sections not found for body evaluation."}

    body_section_blocks = parsed_sections.get("body", []) # Get blocks for 'body' section
    body_plan_text = get_first_text_from_section("body", parsed_sections)

    if body_plan_text == "내용 없음" or not body_plan_text.strip():
        body_heading_exists = any(block.get("type") == "heading_1" and "본문" in block.get("plain_text", "").lower() for block in body_section_blocks)
        if body_heading_exists:
            comment_text = "본문 섹션은 있으나, 세부 기획 내용(인용구 블록)이 없습니다."
        else:
            comment_text = "본문 섹션 자체가 없거나, 인식할 수 있는 본문 관련 내용(예: '본문' 제목)을 찾지 못했습니다."

        logger.info(f"No descriptive quote found in body section. Page comment: '{comment_text}'")
        try:
            tools_by_name["insert_notion_page_comment"].invoke({"page_id": page_id, "text": comment_text})
            return {"body_evaluation_comment": comment_text, "body_review_status": "change_requested", "error_message": None}
        except Exception as e:
            logger.error(f"Error adding page comment for missing body description: {str(e)}")
            return {"error_message": f"Error adding page comment for body: {str(e)}", "body_review_status": "change_requested"}

    logger.info(f"Body plan text found for evaluation: '{body_plan_text[:100]}...'")

    try:
        system_prompt_template = prompt_manager.get_prompt("korean-youtube-title-evaluation-prompt")
        try:
            system_prompt_content = system_prompt_template.format()
        except AttributeError:
            system_prompt_content = str(system_prompt_template)
        
        human_message_content = (
            f"평가할 본문 기획: \n\"{body_plan_text}\"\n\n"
            f"이 본문 기획이 유튜브 영상의 평균 시청 지속 시간을 높이는 데 도움이 될지 평가해주세요. \n"
            f"평가 상태를 'approved' 또는 'change_requested'로 지정해주세요."
        )
        
        messages = [
            SystemMessage(content=system_prompt_content.messages[0]["content"] if hasattr(system_prompt_content, 'messages') and system_prompt_content.messages else system_prompt_content),
            HumanMessage(content=human_message_content),
        ]
        
        # LLM now returns a ReviewOutput object directly
        structured_response: ReviewOutput = llm.invoke(messages)
        
        evaluation_comment = structured_response.comment
        review_status = structured_response.status

        if review_status not in ["approved", "change_requested"]:
            logger.warning(f"Invalid status '{review_status}' received from LLM for body. Defaulting to 'change_requested'.")
            review_status = "change_requested"

        logger.info(f"Body plan evaluation by LLM: Comment='{evaluation_comment}', Status='{review_status}'")

        tools_by_name["insert_notion_block_comment"].invoke({"block_id": body_section_blocks[0].get("id"), "text": f"평가: {evaluation_comment}\n상태: {review_status}"})
        logger.info(f"Block comment added to body plan block {body_section_blocks[0].get('id')}")
        return {"body_evaluation_comment": evaluation_comment, "body_review_status": review_status, "error_message": None}
    except Exception as e:
        logger.error(f"Error during body plan evaluation or commenting: {str(e)}", exc_info=True)
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
