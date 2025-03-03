from datetime import datetime
from enum import Enum
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLAlchemyEnum, select, Double
import uuid

Base = declarative_base()

class TaskStatus(Enum):
    CREATED = "created"
    PENDING = "pending"  # Deprecated
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"

class TaskType(Enum):
    PARSING_PDF = "parsing_pdf"
    PARSING_FOLDER = "parsing_folder"

class BackgroundTask(Base):
    __tablename__ = 'background_task'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    type = Column(SQLAlchemyEnum(TaskType))
    status = Column(SQLAlchemyEnum(TaskStatus))
    payload = Column(String)
    progress = Column(Double)
    create_at = Column(Integer)
    update_at = Column(Integer)

