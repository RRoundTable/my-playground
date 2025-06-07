import json
from typing import List, Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    func
)
from sqlalchemy.orm import declarative_base, relationship

# --- 기본 설정 ---
# 모든 모델이 상속받을 기본 클래스 생성
Base = declarative_base()


# --- User 모델 정의 ---
class User(Base):
    """
    사용자 정보를 관리하는 모델
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True, index=True)
    role = Column(String(20), nullable=False, server_default='user')
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    deleted_at = Column(DateTime(timezone=True), nullable=True) # Soft delete를 위함

    # Relationship: User는 여러 Feedback을 가질 수 있음
    feedbacks = relationship("Feedback", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}', email='{self.email}')>"


# --- Feedback 모델 정의 ---
class Feedback(Base):
    """
    특정 과제(Homework)에 대한 사용자(User)의 제출물과 피드백을 관리하는 모델
    """
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    content = Column(Text, nullable=False) # 사용자가 제출한 과제 내용
    feedback = Column(Text, nullable=True) # 관리자/멘토가 작성한 피드백
    status = Column(String(20), nullable=False, server_default='submitted', index=True)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Foreign Keys: User, Homework 테이블과의 관계 설정
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    thread_id = Column(String(255), ForeignKey("homework.thread_id"), nullable=False)
    
    # Relationships: Feedback은 하나의 User와 하나의 Homework에 속함
    user = relationship("User", back_populates="feedbacks")
    homework = relationship("Homework", back_populates="feedbacks")

    def __repr__(self):
        return f"<Feedback(id={self.id}, user_id={self.user_id}, homework_id={self.homework_id}, status='{self.status}')>"

# --- 사용 예시: 데이터베이스에 테이블 생성 ---
if __name__ == "__main__":
    # SQLite 데이터베이스 엔진 생성 (메모리 또는 파일)
    # engine = create_engine("sqlite:///:memory:") # 테스트용 인-메모리 DB
    engine = create_engine("sqlite:///community_app.db") # 파일 기반 DB

    # Base에 정의된 모든 테이블을 데이터베이스에 생성
    Base.metadata.create_all(engine)

    print("데이터베이스 테이블 'users', 'homework', 'feedback'이 성공적으로 생성되었습니다.")
    print("DB 파일: community_app.db")