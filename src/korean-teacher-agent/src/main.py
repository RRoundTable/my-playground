from fastapi import FastAPI, Depends, HTTPException, status as http_status, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
import logging
import json
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

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

class RowData(BaseModel):
    id: int
    initial_request: str
    homework_topic: str | None = None
    homework_writing_type: str | None = None
    homework_keywords: str | None = None
    detailed_homework: str | None = None
    error_message: str | None = None
    status: str
    created_at: str
    updated_at: str

class RequestData(BaseModel):
    table_id: str
    table_name: str
    rows: list[RowData]

class HomeworkCreationRequest(BaseModel):
    type: str
    id: str
    data: RequestData

    class Config:
        from_attributes = True



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
    request_data: HomeworkCreationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db)
):
    # Extract the initial request from the data structure
    if not request_data.data.rows:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="No rows provided in the request data"
        )
    
    row_data = request_data.data.rows[0]
    
    # Check if homework entry exists
    result = await db.execute(select(Homework).where(Homework.id == row_data.id))
    existing_homework = result.scalar_one_or_none()
    
    if existing_homework:
        # Update existing homework
        existing_homework.initial_request = row_data.initial_request
        existing_homework.status = "requested"
        homework_entry = existing_homework
        logger.info(f"Updating existing homework entry with ID {row_data.id}")
    else:
        # Raise error if homework doesn't exist
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Homework with ID {row_data.id} not found"
        )
    
    db.add(homework_entry)
    await db.commit()
    await db.refresh(homework_entry)

    # Start background task
    background_tasks.add_task(
        create_writing_homework_background_task,
        homework_entry.initial_request,
        homework_entry.id,
        db
    )
    
    logger.info(f"Homework ID {homework_entry.id} 요청 수신. 숙제 생성 작업 백그라운드에서 시작됨.")
    return HomeworkResponse.from_orm_model(homework_entry)