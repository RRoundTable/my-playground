import datetime
import json
from typing import List, Optional
import uuid

from pydantic import BaseModel, Field as PydanticField # SQLAlchemy의 Column과 구분
from sqlalchemy import Column, Integer, String, Text, DateTime, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from .db_setup import async_engine

Base = declarative_base()

# --- Homework 모델 정의 ---
class Homework(Base):
    __tablename__ = "homework"

    id = Column(UUID, primary_key=True, index=True, default=uuid.uuid4)
    initial_request = Column(Text, nullable=False)
    homework_topic = Column(String(255), nullable=True)
    homework_writing_type = Column(String(100), nullable=True)
    homework_keywords = Column(Text, nullable=True) # JSON string
    detailed_homework = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default="requested", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<Homework(id={self.id}, initial_request='{self.initial_request[:30]}...', status='{self.status}')>"

    def set_keywords(self, keywords: List[str]):
        self.homework_keywords = json.dumps(keywords, ensure_ascii=False)

    def get_keywords(self) -> Optional[List[str]]:
        if self.homework_keywords:
            return json.loads(self.homework_keywords)
        return None

# --- 데이터베이스 테이블 생성 함수 (비동기) ---
async def create_db_and_tables_async():
    import os
    from pathlib import Path

    # Create database directory if it doesn't exist
    db_dir = Path("./database")
    db_dir.mkdir(exist_ok=True)

    async with async_engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all) # 필요시 기존 테이블 삭제
        await conn.run_sync(Base.metadata.create_all)

class HomeworkBase(BaseModel):
    initial_request: str
    homework_topic: Optional[str] = None
    homework_writing_type: Optional[str] = None
    keywords_list: Optional[List[str]] = PydanticField(default=None, alias="homework_keywords_list") # API 요청/응답 시 사용
    detailed_homework: Optional[str] = None
    error_message: Optional[str] = None
    status: str = "requested"

    class Config:
        populate_by_name = True # alias 사용 허용

class HomeworkCreate(HomeworkBase):
    pass

class HomeworkUpdate(BaseModel): # 부분 업데이트를 위한 스키마
    initial_request: Optional[str] = None
    homework_topic: Optional[str] = None
    homework_writing_type: Optional[str] = None
    keywords_list: Optional[List[str]] = PydanticField(default=None, alias="homework_keywords_list")
    detailed_homework: Optional[str] = None
    error_message: Optional[str] = None
    status: Optional[str] = None


class HomeworkResponse(HomeworkBase):
    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime
    # DB에서 읽어올 때는 homework_keywords (JSON 문자열)를 keywords_list로 변환하여 보여줄 수 있음
    # 또는 get_keywords()를 활용하여 스키마에서 처리

    @classmethod
    def from_orm_model(cls, homework_model: Homework) -> "HomeworkResponse":
        data = homework_model.__dict__ # Convert ORM model to dict
        data["keywords_list"] = homework_model.get_keywords() # Populate keywords_list
        return cls(**data)

    class Config:
        from_attributes = True # SQLAlchemy 모델과 호환 (v2 Pydantic)
        # orm_mode = True # (v1 Pydantic)