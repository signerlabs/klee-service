from sqlalchemy.ext.declarative import declarative_base
from enum import Enum
import uuid
from sqlalchemy import Column, Integer, String, DateTime, Enum as SQLAlchemyEnum, Float, JSON, Text
from sqlalchemy import Column, String, Float

Base = declarative_base()


class ChatMessageStatus(str, Enum):
    sending = "sending"
    sent = "sent"
    failed = "failed"
    done = "done"


class ChatMessage(Base):
    __tablename__ = 'chat_message'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    conversation_id = Column("conversationId", String)
    question = Column(String)
    answer = Column(String)
    status = Column(String, nullable=False)
    create_at = Column("createAt", Float)
    update_at = Column("updateAt", Float)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.status:
            self.status = self.status.value if isinstance(self.status, ChatMessageStatus) else self.status


class ChatConversation(Base):
    __tablename__ = 'chat_conversation'

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    latestMessage = Column(Text, nullable=False)
    knowledgeIds = Column(JSON, nullable=False)
    noteIds = Column(JSON, nullable=False)
    fileIds = Column(JSON, nullable=False)
    createAt = Column(Float, nullable=False)
    updateAt = Column(Float, nullable=False)
    systemPrompt = Column(Text, nullable=False)
    replyLanguage = Column(String, nullable=False)
    llmId = Column(String, nullable=False)
    knowledgeContent = Column(String)

# 删除重复的定义
