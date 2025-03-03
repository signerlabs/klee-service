from typing import Optional
from pydantic import BaseModel

class EmbeddingFileRequest(BaseModel):
    path: str
    knowledge_id: str
    file_id: str
    file_type: str
    conversation_id: Optional[str] = None

class EmbeddingFolderRequest(BaseModel):
    folder: str  # folder path
    knowledge_id: str
    # file_id: str
    conversation_id: Optional[str] = None

class DeleteFileRequest(BaseModel):
    knowledge_id: str
    file_id: str

class EmbeddingNoteRequest(BaseModel):
    note_id: str
