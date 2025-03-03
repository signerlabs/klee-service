import uuid

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Float, Double, Boolean
from dataclasses import dataclass
# from sqlalchemy.orm import declarative_base

Base = declarative_base()


@dataclass
class ChatMessage(Base):
    __tablename__ = 'llama_remote_mode'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    name = Column(String, default="", nullable=False)
    description = Column(String, default="", nullable=False)
    api_key = Column(String, default="", nullable=False)
    base_url = Column(String, nullable=False)
    provider_id = Column(Double)
    provider = Column(String, nullable=False)
    # 本地后端所需要用到的


    # 新加入
    create_at = Column(Double)
    update_at = Column(Double)
    delete_at = Column(Double)
