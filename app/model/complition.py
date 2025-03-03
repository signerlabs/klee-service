
from typing import List
from pydantic import BaseModel

class Message(BaseModel):
    role:str
    content:str

class CompletionOption(BaseModel):
    stream: bool | None = False
    message: str
    conversation_id: str

class ConversationTitleRequest(BaseModel):
    conversation_id: str

class StopChatRequest(BaseModel):
    conversation_id: str