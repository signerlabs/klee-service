# coding:utf8
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
import uuid
from sqlalchemy import Column, String, Integer, Double, Boolean
from sqlalchemy import Enum as SQLAlchemyEnum

from app.model.base import Base
from pydantic import BaseModel


class KnowledgeCat(Enum):
    """
    knowledge category 枚举
    """
    FILE = "FILE"
    FOLDER = "FOLDER"


class EmbedStatus(Enum):
    """
    knowledge embed status 枚举
    """
    EMBEDDING = "EMBEDDING"
    EMBEDDED = "EMBEDDED"


class FilePathType(Enum):
    """
    文件路径类型
    """
    RELATIVE = "RELATIVE"
    ABSOLUTE = "ABSOLUTE"

@dataclass
class Knowledge(Base):
    """
    客户端创建的 knowledge 表
    """
    __tablename__ = 'knowledge'

    id: str = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4()),
        index=True
    )  # lowercase uuid
    timeStamp: float = Column(Double, nullable=False)
    create_at = Column(Double)
    update_at = Column(Double)
    delete_at = Column(Double)

    title: str = Column(String, nullable=False)
    icon: str = Column(String, default="", nullable=False)
    description: str = Column(String, default="", nullable=False)

    category: KnowledgeCat = Column(
        SQLAlchemyEnum(KnowledgeCat), default=KnowledgeCat.FILE, nullable=False
    )
    isPin: bool = Column(Boolean, default=False, nullable=False)  # is pinned to top
    folder_path: str = Column(String, default="", nullable=False)
    embed_status: str = Column(
        SQLAlchemyEnum(EmbedStatus), default=EmbedStatus.EMBEDDING, nullable=False
    )  # embedding|embedded
    parent_id: str = Column(String, default="", nullable=True)
    local_mode: bool = Column(Boolean, default="", nullable=True)

@dataclass
class File(Base):
    """
    File table
    """
    __tablename__ = 'file'

    id: str = Column(
        String, primary_key=True, default=lambda: str(uuid.uuid4()).upper(),
        index=True
    )  # 对于 RELATIVE 是大写的uuid, 对于 ABSOLUTE 是 knowledgeId+path 的 digest

    knowledgeId: str = Column(String, nullable=False)  # 大写的uuid
    conversationId: str = Column(String)

    name: str = Column(String, default="", nullable=False)
    format: str = Column(String, default="", nullable=False)

    size: int = Column(Integer, default=0, nullable=False)  # 文件大小，单位字节
    os_mtime: float = Column(Double, default=0, nullable=False)  # 文件修改时间戳
    uploaded: float = Column(
        Integer, default=lambda: int(datetime.now().timestamp()), nullable=False
    )

    path: str = Column(String, default="", nullable=False)
    path_type: str = Column(
        SQLAlchemyEnum(FilePathType), default=FilePathType.RELATIVE,
        nullable=False
    )

    # 新加入
    create_at = Column(Double)
    update_at = Column(Double)
    delete_at = Column(Double)


def abs_file_id(knowledge_id: str, path: str) -> str:
    """
    ABSOLUTE 类型的 file id 生成函数
    """
    s = f"{knowledge_id}{path}"
    return hashlib.md5(s.encode('utf-8')).hexdigest()

class KnowledgeCreate(BaseModel):
    title: str
    icon: str = ""
    description: str = ""
    category: KnowledgeCat = KnowledgeCat.FILE
    isPin: bool = False
    folder_path: str = ""


class KnowledgeEmbedding(BaseModel):
    temp_file_urls: list[str]
    vector_store_urls: list[str]


class KnowledgeResponse(BaseModel):
    id: str
    timeStamp: float
    title: str
    icon: str
    description: str
    category: KnowledgeCat
    isPin: bool
    folder_path: str
    embed_status: EmbedStatus
    create_at: float
    update_at: float

    class Config:
        from_attributes = True