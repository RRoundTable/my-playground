from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict
import logging
import json
from contextlib import asynccontextmanager

from src.agents.writing_homework_agent import wrting_homework_agent
from src.database.homework import create_db_and_tables_async

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



@app.get("/")
async def root():
    return {"message": "Welcome to Korean Teacher Agent API"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }



