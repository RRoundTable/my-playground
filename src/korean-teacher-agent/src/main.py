from fastapi import FastAPI, Depends, HTTPException, status as http_status, BackgroundTasks
from pydantic import BaseModel
from typing import Dict
import logging
import json
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession

from src.agents.writing_homework_agent import wrting_homework_agent

from src.database.homework import create_db_and_tables_async, HomeworkResponse, Homework
from src.database.db_setup import get_async_db
from src.services.writing_homework import create_writing_homework_background_task


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# lifespan 컨텍스트 매니저 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 실행될 코드
    logger.info("애플리케이션 시작 - 데이터베이스 및 테이블 초기화...")
    await create_db_and_tables_async() # DB 및 테이블 생성
    logger.info("데이터베이스 및 테이블 초기화 완료.")
    
    yield 
    logger.info("애플리케이션 종료 중...")


app = FastAPI(
    title="Korean Teacher Agent",
    description="A dummy FastAPI application for Korean Teacher Agent",
    version="1.0.0",
    lifespan=lifespan
)

class HealthResponse(BaseModel):
    status: str
    version: str

class HomeworkCreationRequest(BaseModel):
    initial_request: str



@app.get("/")
async def root():
    return {"message": "Welcome to Korean Teacher Agent API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }

@app.post("/writing-homework", response_model=HomeworkResponse, status_code=http_status.HTTP_202_ACCEPTED)
async def create_writing_homework(
    request_data: HomeworkCreationRequest, # Pydantic 모델로 요청 본문 받기
    background_tasks: BackgroundTasks,    # 백그라운드 작업 주입
    db: AsyncSession = Depends(get_async_db)
):
    # 1. 요청 수신 즉시 Homework 테이블에 row 생성
    new_homework_entry = Homework(
        initial_request=request_data.initial_request,
        status="requested" # 초기 상태는 'requested'
    )
    db.add(new_homework_entry)
    await db.commit()
    await db.refresh(new_homework_entry) # DB에서 할당된 ID 등을 가져오기 위해

    # 백그라운드 작업 시작
    background_tasks.add_task(
        create_writing_homework_background_task,
        new_homework_entry.initial_request, # 사용자 요청 원문
        new_homework_entry.id,              # 생성된 숙제 항목의 ID
        db                                  # 현재 요청의 DB 세션을 전달 (주의사항 확인)
                                            # 더 안전한 방법: 백그라운드 함수 내에서 새 세션 생성
                                            # 예: async def mock_homework_agent_logic(req, id_):
                                            #         async with AsyncSessionLocal() as session:
                                            #              # session 사용
    )
    
    print(f"Homework ID {new_homework_entry.id} 요청 수신. 숙제 생성 작업 백그라운드에서 시작됨.")
    return HomeworkResponse.from_orm_model(new_homework_entry)