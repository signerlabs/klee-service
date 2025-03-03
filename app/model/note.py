from sqlalchemy import Column, String, Boolean, Enum
from sqlalchemy.types import Double
from sqlalchemy.ext.declarative import declarative_base
from enum import Enum as PyEnum
from pydantic import BaseModel
from sqlalchemy import create_engine

# engine = create_engine("sqlite+aiosqlite:///F:/sqlite/test_db.db")
Base = declarative_base()


class NoteType(PyEnum):
    note = "note"
    folder = "folder"


class NoteStatus(PyEnum):
    normal = "normal"
    deleted = "deleted"
    draft = "draft"


class Note(Base):
    __tablename__ = 'note'

    id = Column(String, primary_key=True)
    folder_id = Column(String)
    title = Column(String)
    content = Column(String)
    type = Column(Enum(NoteType))
    status = Column(Enum(NoteStatus))
    is_pin = Column(Boolean)
    create_at = Column(Double)
    update_at = Column(Double)
    delete_at = Column(Double, default=0)
    html_content = Column(String, default="")
    local_mode = Column(Boolean, default=False, nullable=True)

    def __init__(self, id, folder_id, title, content, type, status, is_pin, create_at, update_at, delete_at=0,
                 html_content="", local_mode=False):
        self.id = id
        self.folder_id = folder_id
        self.title = title
        self.content = content
        self.type = type
        self.status = status
        self.is_pin = is_pin
        self.create_at = create_at
        self.update_at = update_at
        self.delete_at = delete_at
        self.html_content = html_content
        self.local_mode = local_mode

    def __repr__(self):
        return f"<Note(id='{self.id}', title='{self.title}')>"


class CreateNoteRequest(BaseModel):
    id: str
    folder_id: str
    title: str
    content: str
    type: NoteType
    status: NoteStatus
    is_pin: bool
    html_content: str = ""
    temp_file_url: str = ""


class NoteFileListId(BaseModel):
    temp_file_urls: list[str]
    vector_file_urls: list[str]


class NoteResponse(BaseModel):
    id: str
    folder_id: str
    title: str
    content: str
    type: NoteType
    status: NoteStatus
    is_pin: bool
    create_at: float
    update_at: float
    delete_at: float
    html_content: str
    local_mode: bool
