import os
import logging
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.language_models import BaseChatModel # LLM 타입 힌트를 위해 사용

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('writing_homework_agent')

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 확인 (스크립트 실행 시 한 번만)
OPENAI_API_KEY_LOADED = os.getenv("OPENAI_API_KEY") is not None
if not OPENAI_API_KEY_LOADED:
    logger.warning("OpenAI API 키가 .env 파일에 설정되지 않았습니다.")
    # 여기서 exit()를 호출하면 프로그램이 즉시 종료됩니다.
    # main 함수에서 키 유무를 확인하고 처리하도록 변경할 수도 있습니다.

# 2. 에이전트 상태 정의
class AgentState(TypedDict):
    initial_request: str
    homework_idea: Dict[str, Any]
    detailed_homework: str
    error_message: Optional[str] # 오류 메시지는 없을 수도 있으므로 Optional 사용

# 3. 숙제 아이디어 구상 노드용 Pydantic 모델
class HomeworkIdea(BaseModel):
    topic: str = Field(description="숙제의 주제 (한국어)")
    writing_type: str = Field(description="글쓰기 유형 (예: 짧은 이야기, 후기, 일기) (한국어)")
    keywords: List[str] = Field(description="숙제 주제와 관련된 핵심 키워드 (한국어)")


def create_korean_homework_agent() -> StateGraph:
    """
    주어진 LLM을 사용하여 한국어 글쓰기 숙제 생성 LangGraph 에이전트를 생성하고 반환합니다.
    """

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
    # --- 노드 함수 정의 ---
    async def brainstorm_homework_idea_node(state: AgentState):
        logger.info("--- 숙제 아이디어 구상 중 ---")
        parser = JsonOutputParser(pydantic_object=HomeworkIdea)

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "당신은 창의적인 한국어 글쓰기 숙제 아이디어를 제안하는 AI입니다. "
             "사용자의 요청에 따라 숙제의 주제, 글쓰기 유형, 그리고 관련 키워드를 제안해야 합니다. "
             "결과는 반드시 JSON 형식으로 반환하며, 모든 내용은 한국어로 작성되어야 합니다.\n"
             "{format_instructions}"),
            ("human", "{request}")
        ])

        chain = prompt | llm | parser
        try:
            response = await chain.ainvoke({
                "request": f"'{state['initial_request']}' 요청에 대한 창의적인 한국어 글쓰기 숙제 아이디어를 제안해주세요.",
                "format_instructions": parser.get_format_instructions()
            })
            logger.info(f"구상된 아이디어: {response}")
            return {"homework_idea": response, "error_message": None}
        except Exception as e:
            logger.error(f"숙제 아이디어 구상 중 오류: {e}")
            return {"error_message": f"숙제 아이디어 구상 실패: {str(e)}"}

    async def generate_detailed_homework_node(state: AgentState):
        logger.info("--- 상세 숙제 내용 생성 중 ---")
        if state.get("error_message"):
            logger.warning("이전 단계에서 오류가 발생하여 상세 숙제 생성을 건냅니다.")
            return {}
        if not state.get("homework_idea") or not isinstance(state["homework_idea"], dict):
            logger.error("상세 숙제 생성을 위한 아이디어가 없습니다.")
            return {"error_message": "상세 숙제 생성을 위한 아이디어가 없습니다."}

        idea = state["homework_idea"]
        topic = idea.get("topic", "정해지지 않은 주제")
        writing_type = idea.get("writing_type", "일반 글쓰기")
        keywords_str = ", ".join(idea.get("keywords", []))

        prompt = ChatPromptTemplate.from_messages([
            ("system",
            "당신은 한국어 글쓰기 숙제 안내문을 작성하는 AI입니다. "
            "주어진 주제, 글쓰기 유형, 키워드를 바탕으로 학생들에게 제시할 구체적이고 친절한 숙제 안내문을 작성해주세요. "
            "숙제 안내문에는 다음 내용이 포함되어야 합니다:\n"
            "1. 숙제 제목 (흥미를 유발하는 창의적인 제목)\n"
            "2. 명확한 글쓰기 지시사항 (무엇을, 어떻게 써야 하는지)\n"
            "3. 포함되어야 할 내용 또는 고려사항 (필요시 키워드 활용 안내 포함)\n"
            "4. 예상 분량 (5줄 이내)\n" # 안내 문구 수정
            "모든 내용은 한국어로 자연스럽게 작성해주세요.\n" # 줄바꿈 추가
            "예시 숙제 안내문 (학생에게 제시될 글쓰기 과제 부분):\n" # "예시:" 문구 명확화
            "--------------------------------------------------\n"
            "✨ 나의 특별한 생일 이야기 ✨\n\n"
            "친구들, 안녕하세요! 오늘은 여러분의 생일에 대한 이야기를 나눠볼 거예요.\n\n"
            "🎂 생일이 언제예요?\n"
            "🎁 그리고 생일에 받고 싶은 선물이나 하고 싶은 것을 얘기해볼까요?\n"
            "🎉 이미 생일이 지났다면 생일에 했던 일, 받았던 선물을 얘기해볼까요?\n\n"
            "자유롭게 여러분의 생각이나 경험을 표현해주세요.\n"
            "✏️ 5줄 이상으로 작성해봐요 😄\n"
            "--------------------------------------------------"
            ),

            ("human",
            "다음 정보를 바탕으로 상세한 한국어 글쓰기 숙제 안내문을 작성해주세요:\n"
            f"- 주제: {topic}\n"
            f"- 글쓰기 유형: {writing_type}\n"
            f"- 관련 키워드: {keywords_str}\n"
            "학생들이 이해하기 쉽고 흥미를 느낄 수 있도록 작성해주세요. 위 시스템 메시지의 예시 스타일을 참고하여, 특히 글쓰기 과제 부분을 명확하고 친근하게 제시해주세요.")
        ])

        chain = prompt | llm
        try:
            response = await chain.ainvoke({})
            detailed_homework = response.content
            logger.info(f"생성된 숙제:\n{detailed_homework}")
            return {"detailed_homework": detailed_homework, "error_message": None}
        except Exception as e:
            logger.error(f"상세 숙제 내용 생성 중 오류: {e}")
            return {"error_message": f"상세 숙제 내용 생성 실패: {str(e)}"}

    async def handle_error_node(state: AgentState):
        logger.error("--- 오류 발생 ---")
        error_msg = state.get('error_message', "알 수 없는 오류")
        logger.error(f"에러 메시지: {error_msg}")
        return {"error_message": error_msg, "detailed_homework": ""}

    # --- 그래프 구성 ---
    workflow = StateGraph(AgentState)

    workflow.add_node("brainstorm_idea", brainstorm_homework_idea_node)
    workflow.add_node("generate_homework", generate_detailed_homework_node)
    workflow.add_node("error_handler", handle_error_node)

    workflow.set_entry_point("brainstorm_idea")

    workflow.add_conditional_edges(
        "brainstorm_idea",
        lambda x: "error_handler" if x.get("error_message") else "generate_homework",
        {
            "generate_homework": "generate_homework",
            "error_handler": "error_handler"
        }
    )
    workflow.add_conditional_edges(
        "generate_homework",
        lambda x: "error_handler" if x.get("error_message") else END,
        {
            END: END,
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("error_handler", END)

    return workflow.compile()

wrting_homework_agent = create_korean_homework_agent()

if __name__ == "__main__":
    import asyncio

    async def main():
        logger.info("🇰🇷 한국어 글쓰기 숙제 생성 에이전트 (OpenAI) ✏️")

        if not OPENAI_API_KEY_LOADED:
            logger.error("OpenAI API 키가 로드되지 않았습니다. .env 파일을 확인하고 프로그램을 다시 시작해주세요.")
            return
        homework_agent_app = create_korean_homework_agent()

        user_request = input("어떤 종류의 글쓰기 숙제를 만들까요? (예: '나의 꿈에 대한 수필 숙제', '환경 보호 논설문', '자유 주제 시 쓰기')\n> ")

        if not user_request:
            user_request = "창의적인 한국어 글쓰기 숙제를 하나 만들어주세요."
            logger.info(f"사용자 입력이 없어 기본 요청으로 설정: {user_request}")


        initial_state = {
            "initial_request": user_request,
            "homework_idea": {},
            "detailed_homework": "",
            "error_message": None
        }

        logger.info("\n숙제 생성을 시작합니다...")
        final_state = None

        async for event_value in homework_agent_app.astream(initial_state, stream_mode="values"):
            if isinstance(event_value, list):
                current_node_output_state = event_value[-1] if event_value else {}
            else:
                current_node_output_state = event_value

            if current_node_output_state:
                 final_state = current_node_output_state
                 # 각 단계의 상태를 로깅하고 싶다면 다음 줄의 주석을 해제하세요.
                 # logger.debug(f"중간 상태 업데이트: {final_state}")


        logger.info("\n--- 최종 결과 ---")
        if final_state:
            if final_state.get("detailed_homework") and final_state["detailed_homework"].strip():
                logger.info("\n🎉 짜잔! 숙제가 완성되었어요! 🎉\n")
                # 최종 숙제 내용은 logger.info로 출력하면 너무 길어질 수 있으므로 print로 유지하거나,
                # 파일로 저장하는 등의 다른 방식을 고려할 수 있습니다.
                # 여기서는 명확한 구분을 위해 print를 사용합니다.
                print(final_state["detailed_homework"])
            elif final_state.get("error_message"):
                logger.error(f"\n🚨 숙제 생성 중 문제가 발생했습니다: {final_state['error_message']}")
            else:
                logger.warning("\n🤔 숙제가 정상적으로 생성되지 않았거나, 내용이 없습니다.")
                # logger.debug(f"최종 상태: {final_state}")
        else:
            logger.error("\n숙제가 생성되지 않았습니다. (최종 상태 정보를 받지 못함)")

    asyncio.run(main())