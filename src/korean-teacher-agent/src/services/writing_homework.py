# 필요한 모듈 임포트 (이미 임포트 되어 있을 수 있음)
import logging # 로깅 추가
from typing import Dict, Any, Optional # AgentState 타입 힌트와 일치시키기 위해 필요

# --- 이전 코드에서 정의된 요소들 ---
from src.agents.writing_homework_agent import wrting_homework_agent, AgentState # 실제 에이전트 앱과 상태 정의
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.homework import Homework # SQLAlchemy 모델
from sqlalchemy.future import select

# 로거 설정 (함수 외부 또는 모듈 레벨에서 설정)
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # 기본 설정 (필요시 더 상세하게 설정)

async def create_writing_homework_background_task(
    initial_request: str,
    db_homework_id: int,
    db: AsyncSession,
):
    """
    실제 숙제 생성 에이전트 로직을 수행하는 백그라운드 작업 함수.
    """
    logger.info(f"백그라운드 작업 시작: Homework ID {db_homework_id}에 대한 숙제 생성 중...")
    logger.info(f"사용자 초기 요청: '{initial_request}'")

    # 1. DB에서 숙제 항목 조회 및 상태 'processing'으로 업데이트
    result = await db.execute(select(Homework).where(Homework.id == db_homework_id))
    db_homework_item: Optional[Homework] = result.scalar_one_or_none() # 타입 힌트 명시

    if not db_homework_item:
        logger.error(f"오류: Homework ID {db_homework_id}를 찾을 수 없습니다 (백그라운드).")
        return

    db_homework_item.status = "processing"
    db.add(db_homework_item)
    await db.commit()
    await db.refresh(db_homework_item)
    logger.info(f"Homework ID {db_homework_id} 상태를 'processing'으로 업데이트 완료.")

    # 2. 실제 에이전트 로직 호출
    # AgentState 초기화 (필요한 모든 키를 명시적으로 포함)
    initial_agent_state_dict: Dict[str, Any] = {
        "initial_request": initial_request,
        "homework_idea": {},  # 초기에는 비어있음
        "detailed_homework": "",  # 초기에는 비어있음
        "error_message": None  # 초기에는 에러 없음
    }

    final_agent_state_data: Optional[AgentState] = None # AgentState 타입으로 결과 저장

    try:
 
        logger.info(f"LangGraph 에이전트 호출 시작 (Homework ID: {db_homework_id})...")
        
        # astream 메서드를 사용하여 에이전트 실행 및 최종 상태 수집
        async for event_value in wrting_homework_agent.astream(initial_agent_state_dict, stream_mode="values"):
            # LangGraph의 stream_mode="values"는 각 노드 반환값 또는 최종 AgentState를 스트리밍합니다.
            # 일반적으로 마지막 이벤트가 최종 상태를 포함합니다.
            if isinstance(event_value, list): # 최신 LangGraph는 리스트로 마지막 상태를 포함하여 반환 가능
                current_node_output_state = event_value[-1] if event_value else {}
            else: # 단일 상태 객체 반환 시
                current_node_output_state = event_value
            
            if current_node_output_state: # None이 아닐 경우에만 업데이트
                 final_agent_state_data = current_node_output_state # AgentState 타입으로 캐스팅될 수 있음

        logger.info(f"LangGraph 에이전트 호출 완료 (Homework ID: {db_homework_id}).")

        # 최종 상태가 AgentState 타입의 딕셔너리인지 확인하고 사용
        if not isinstance(final_agent_state_data, dict):
             logger.error(f"에이전트가 유효한 딕셔너리 형태의 최종 상태를 반환하지 않았습니다: {type(final_agent_state_data)}")
             # 필요한 경우 AgentState 타입으로 변환 시도 또는 오류 처리
             # 여기서는 final_agent_state_data가 AgentState의 키를 가진 딕셔너리라고 가정
             if final_agent_state_data is None: # final_agent_state_data가 None일 경우를 대비
                final_agent_state_data = {} # 빈 딕셔너리로 초기화하여 아래 로직에서 KeyError 방지

    except Exception as e:
        logger.error(f"에이전트 실행 중 오류 발생 (Homework ID: {db_homework_id}): {e}", exc_info=True)
        # final_agent_state_data가 AgentState 구조를 따르도록 오류 정보 설정
        final_agent_state_data = AgentState( # 오류 발생 시 AgentState 구조로 초기화
            initial_request=initial_request,
            homework_idea={},
            detailed_homework="",
            error_message=f"에이전트 실행 오류: {str(e)}"
        )

    # 3. 에이전트 결과를 바탕으로 DB 업데이트
    if final_agent_state_data: # final_agent_state_data가 None이 아니거나 빈 딕셔너리가 아님을 보장
        # AgentState의 키를 사용하여 안전하게 접근
        homework_idea = final_agent_state_data.get("homework_idea", {})
        db_homework_item.homework_topic = homework_idea.get("topic")
        db_homework_item.homework_writing_type = homework_idea.get("writing_type")
        
        keywords = homework_idea.get("keywords")
        if keywords: # keywords가 None이 아니고 비어있지 않은 리스트일 수 있음
            db_homework_item.set_keywords(keywords)
        
        db_homework_item.detailed_homework = final_agent_state_data.get("detailed_homework", "")
        db_homework_item.error_message = final_agent_state_data.get("error_message")
        
        if db_homework_item.error_message:
            db_homework_item.status = "error"
            logger.warning(f"숙제 생성 중 오류 발생 (Homework ID: {db_homework_id}): {db_homework_item.error_message}")
        else:
            db_homework_item.status = "generated"
            logger.info(f"숙제 성공적으로 생성됨 (Homework ID: {db_homework_id})")
    else:
        # 이 경우는 위 try-except 블록에서 final_agent_state_data가 초기화되므로 거의 발생하지 않음
        logger.error(f"에이전트가 최종 상태 데이터를 반환하지 못했습니다 (Homework ID: {db_homework_id}).")
        db_homework_item.error_message = "에이전트가 결과를 반환하지 못했습니다."
        db_homework_item.status = "error"

    db.add(db_homework_item)
    await db.commit()
    await db.refresh(db_homework_item) # DB에서 최종 상태 가져오기 (선택적)
    logger.info(f"백그라운드 작업 완료: Homework ID {db_homework_id} 숙제 생성 및 DB 업데이트 완료. 최종 상태: {db_homework_item.status}")