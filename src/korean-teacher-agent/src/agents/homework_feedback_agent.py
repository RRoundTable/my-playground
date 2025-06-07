import os
import logging
import asyncio
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- 기본 설정 ---

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('korean_correction_agent')

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 확인
OPENAI_API_KEY_LOADED = os.getenv("OPENAI_API_KEY") is not None
if not OPENAI_API_KEY_LOADED:
    logger.warning("OpenAI API 키가 .env 파일에 설정되지 않았습니다. .env 파일을 생성하고 OPENAI_API_KEY='your_key' 형식으로 키를 입력해주세요.")

# --- Pydantic 모델 및 에이전트 상태 정의 ---

class Correction(BaseModel):
    """개별 첨삭 항목을 정의하는 모델"""
    original: str = Field(description="학생의 원본 문장 또는 어구")
    corrected: str = Field(description="첨삭된 문장 또는 어구")
    explanation: str = Field(description="첨삭 이유에 대한 상세한 설명 (문법, 어휘 등)")

class CorrectionResult(BaseModel):
    """LLM의 첨삭 결과를 구조화하기 위한 Pydantic 모델"""
    grammar_corrections: List[Correction] = Field(description="문법적으로 어색하거나 틀린 부분에 대한 첨삭 리스트")
    vocabulary_suggestions: List[Correction] = Field(description="더 나은 어휘나 표현에 대한 제안 리스트")
    general_feedback: str = Field(description="글 전체에 대한 종합적인 평가 및 격려 메시지")

class AgentState(TypedDict):
    """에이전트의 상태를 관리하는 객체"""
    homework_prompt: str             # 숙제의 주제 또는 요구사항
    student_submission: str          # 학생이 제출한 글
    correction_result: Optional[CorrectionResult] # 최종 첨삭 결과
    error_message: Optional[str]     # 오류 메시지

# --- LangGraph 노드 함수 정의 ---

def create_korean_correction_agent() -> StateGraph:
    """
    한국어 글쓰기 첨삭 LangGraph 에이전트를 생성하고 반환합니다.
    """
    # LLM 모델 초기화 (GPT-4o가 한국어 교정에 더 강한 성능을 보일 수 있습니다)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    async def correction_node(state: AgentState):
        """숙제와 학생의 제출물을 바탕으로 문법, 단어를 첨삭하는 노드"""
        logger.info("--- 학생 제출물 첨삭 시작 ---")
        parser = JsonOutputParser(pydantic_object=CorrectionResult)

        prompt = ChatPromptTemplate.from_template(
            """당신은 한국어 선생님입니다. 학생이 제출한 글을 아래의 숙제 요구사항에 맞게 첨삭해주세요. 문법 오류, 어색한 어휘 사용을 중심으로 수정하고, 각 수정 사항에 대해 친절하고 이해하기 쉬운 설명을 덧붙여주세요. 마지막으로 글 전체에 대한 종합적인 피드백을 남겨주세요.

숙제 요구사항:
{homework_prompt}

학생이 제출한 글:
{student_submission}

{format_instructions}
"""
        )

        chain = prompt | llm | parser

        try:
            # state에서 필요한 정보 추출
            homework_prompt = state['homework_prompt']
            student_submission = state['student_submission']

            response = await chain.ainvoke({
                "homework_prompt": homework_prompt,
                "student_submission": student_submission,
                "format_instructions": parser.get_format_instructions(),
            })

            logger.info("첨삭이 성공적으로 완료되었습니다.")
            return {"correction_result": response, "error_message": None}
        except Exception as e:
            logger.error(f"첨삭 진행 중 오류 발생: {e}")
            return {"error_message": f"첨삭 실패: {str(e)}"}

    async def handle_error_node(state: AgentState):
        """오류 발생 시 처리하는 노드"""
        logger.error("--- 오류 발생 ---")
        error_msg = state.get('error_message', "알 수 없는 오류")
        logger.error(f"에러 메시지: {error_msg}")
        # 오류 발생 시 첨삭 결과는 초기화
        return {"correction_result": None}

    # --- 그래프 구성 ---
    workflow = StateGraph(AgentState)

    workflow.add_node("corrector", correction_node)
    workflow.add_node("error_handler", handle_error_node)

    workflow.set_entry_point("corrector")

    # corrector 노드 실행 후 조건에 따라 분기
    workflow.add_conditional_edges(
        "corrector",
        lambda x: "error_handler" if x.get("error_message") else END,
        {
            END: END,
            "error_handler": "error_handler"
        }
    )
    workflow.add_edge("error_handler", END)

    # 그래프 컴파일
    return workflow.compile()


# --- 에이전트 실행 ---

async def main():
    """메인 실행 함수"""
    logger.info("🇰🇷 한국어 글쓰기 첨삭 에이전트 📝")

    if not OPENAI_API_KEY_LOADED:
        logger.error("OpenAI API 키가 로드되지 않았습니다. 프로그램을 종료합니다.")
        return

    # 예제 데이터 정의
    homework = "여러분이 가장 좋아하는 계절에 대해 설명하는 글을 써보세요. 그 계절을 왜 좋아하는지, 그 계절에는 주로 무엇을 하는지에 대한 내용을 포함하여 5문장 이상으로 작성하세요."
    student_submission = "내가 가장 좋은 계절은 가을입니다. 왜냐하면 날씨가 너무 덥지않고 시원하다. 나는 가을에 공원에서 책을 읽는 것을 좋아하고 친구랑 같이 놀아요. 단풍잎의 색깔이 예뻐요. 그래서 나는 가을을 기다려진다."

    logger.info("\n--- 숙제 내용 ---")
    print(homework)
    logger.info("\n--- 학생 제출안 ---")
    print(student_submission)

    # LangGraph 에이전트 생성
    correction_agent_app = create_korean_correction_agent()

    # 초기 상태 설정
    initial_state = {
        "homework_prompt": homework,
        "student_submission": student_submission,
        "correction_result": None,
        "error_message": None
    }

    logger.info("\n첨삭을 시작합니다...")

    final_state = None
    # astream_events는 더 상세한 중간 과정을 보여주지만, 여기서는 최종 결과만 필요하므로 astream을 사용합니다.
    async for state in correction_agent_app.astream(initial_state):
        final_state = state

    logger.info("\n--- 최종 첨삭 결과 ---")
    if final_state and 'corrector' in final_state:
        result = final_state['corrector'].get("correction_result")
        if result:
            print("\n✅ 문법 교정")
            for item in result['grammar_corrections']:
                print(f"  - 원문: {item['original']}")
                print(f"  - 교정: {item['corrected']}")
                print(f"  - 설명: {item['explanation']}\n")

            print("\n✅ 어휘 및 표현 제안")
            for item in result['vocabulary_suggestions']:
                print(f"  - 원문: {item['original']}")
                print(f"  - 제안: {item['corrected']}")
                print(f"  - 설명: {item['explanation']}\n")

            print("\n✅ 종합 평가")
            print(f"  {result['general_feedback']}")
            print("\n🎉 첨삭이 완료되었습니다! 🎉")

        elif final_state['corrector'].get("error_message"):
            error_msg = final_state['corrector']["error_message"]
            logger.error(f"\n🚨 첨삭 중 문제가 발생했습니다: {error_msg}")
        else:
             logger.warning("\n🤔 첨삭 결과가 생성되지 않았습니다.")
    else:
        logger.error("\n첨삭이 정상적으로 실행되지 않았습니다.")


if __name__ == "__main__":
    asyncio.run(main())