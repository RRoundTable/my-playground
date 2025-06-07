from fastapi import FastAPI, Depends, HTTPException, status as http_status, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import json
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.agents.writing_homework_agent import wrting_homework_agent

from src.database.homework import create_db_and_tables_async, HomeworkResponse, Homework
from src.database.db_setup import get_async_db
from src.services.writing_homework import create_writing_homework_background_task
from dotenv import load_dotenv
from src.services.heartbeat_service import send_homework_to_heartbeat
import os
import uuid
load_dotenv()
from phoenix.otel import register

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
phoenix_secret = os.getenv("PHOENIX_API_KEY", "")


logger.info(f"Phoenix endpoint: {phoenix_endpoint}")
logger.info(f"Phoenix API key configured: {bool(phoenix_secret)}")

 
# register 함수로 설정 초기화 (API 키가 있으면 자동으로 Authorization 헤더 추가)
tracer_provider = register(
    project_name="korean-teacher-agent",
    protocol="grpc",
    endpoint=phoenix_endpoint,
    headers={"Authorization": f"Bearer {phoenix_secret}"} if phoenix_secret else {},
    auto_instrument=True
)   



# lifespan 컨텍스트 매니저 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 애플리케이션 시작 시 실행될 코드
    logger.info("애플리케이션 시작 - 데이터베이스 및 테이블 초기화...")
    # await create_db_and_tables_async() # DB 및 테이블 생성

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

class Button(BaseModel):
    type: str
    label: str
    fk_webhook_id: str

class RowData(BaseModel):
    id: int
    initial_request: str
    homework_topic: str | None = None
    homework_writing_type: str | None = None
    homework_keywords: str | None = None
    detailed_homework: str | None = None
    error_message: str | None = None
    status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

class RequestData(BaseModel):
    table_id: str
    table_name: str
    rows: list[RowData]

class HomeworkRequest(BaseModel):
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
    request_data: HomeworkRequest,
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


@app.post("/send-writing-homework", response_model=HomeworkResponse, status_code=http_status.HTTP_200_OK)
async def send_writing_homework(
    homework_request: HomeworkRequest,
    db: AsyncSession = Depends(get_async_db)
):
    logger.info(f"Homework request: {homework_request}")
    
    # Extract the homework ID from the structure
    if not homework_request.data.rows:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="No homework data found in the request"
        )
    
    homework_id = homework_request.data.rows[0].id
    logger.info(f"Homework ID: {homework_id}")
    
    result = await db.execute(select(Homework).where(Homework.id == homework_id))
    homework = result.scalar_one_or_none()
    if not homework:
        raise HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=f"Homework with ID {homework_id} not found")
    
    # Send to Heartbeat asynchronously
    channel_id = os.getenv("HEARTBEAT_CHANNEL_ID", "8478bca8-9818-405e-89b8-81f6bb0aaf8e")
    from_uuid = os.getenv("HEARTBEAT_FROM_UUID", "1a54fcc7-3580-46e0-9edd-017a4e2a139e")
    response = await send_homework_to_heartbeat(homework, channel_id, from_uuid)


    if response is None:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send hmework to Heartbeat channel"
        )

    homework.status = "sended"
    homework.homework_url = response.get("url")
    db.add(homework)
    await db.commit()
    await db.refresh(homework)

    return HomeworkResponse.from_orm_model(homework)