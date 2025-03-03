import uuid

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, Double, Boolean
from dataclasses import dataclass
# from sqlalchemy.orm import declarative_base

Base = declarative_base()


@dataclass
class ChatMessage(Base):
    __tablename__ = 'llama_chat_message'
    id: str = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    role: str = Column(String, default="", nullable=False)
    content: str = Column(String, default="", nullable=False)
    conversation_id: str = Column(String, default="", nullable=False)
    create_time = Column(Double, nullable=False)
    modify_time = Column(Double)
    status: str = Column(String, default="success", nullable=False)
    error_message: str = Column(String, default="success", nullable=False)
    # 新加入
    create_at = Column(Double)
    update_at = Column(Double)
    delete_at = Column(Double)


@dataclass
class Conversation(Base):
    __tablename__ = "llama_chat_conversation"
    # __table_args__ = {'autoload': True, 'extend_existing': True}

    id: str = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    knowledge_ids: str = Column(String, default="", nullable=False)
    note_ids: str = Column(String, default="", nullable=False)
    title: str = Column(String, default="", nullable=False)
    create_time: float = Column(Double, default=0, nullable=False)
    is_pin: bool = Column(Boolean, default=False, nullable=False)
    # 扩充基础配置在这里
    model_name: str = Column(String, default="")
    local_mode: str = Column(Boolean, default=False, nullable=False)
    provider_id: str = Column(String, default="", nullable=False)
    model_id: str = Column(String, default="", nullable=False)
    language_id: str = Column(String, default="", nullable=False)
    system_prompt: str = Column(String, default="", nullable=False)
    model_path: str = Column(String, default="", nullable=False)
    # 新加入
    create_at = Column(Double)
    delete_at = Column(Double)
    update_at = Column(Double)

