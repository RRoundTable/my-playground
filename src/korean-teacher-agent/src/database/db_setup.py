import datetime
import json
from typing import List, Optional, AsyncGenerator

from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

# --- SQLAlchemy 비동기 설정 ---
ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./database/sqlite.db" # 비동기 드라이버 사용

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    # echo=True, # SQL 쿼리 로깅이 필요하면 주석 해제
)

# 비동기 세션 메이커
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    expire_on_commit=False, # 비동기 환경, 특히 FastAPI와 함께 사용할 때 권장
    class_=AsyncSession # AsyncSession 클래스 명시
)

Base = declarative_base()

# --- FastAPI 의존성: 비동기 DB 세션 제공 ---
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
