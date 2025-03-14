import logging

from pydantic import BaseModel
from typing import Optional, List
from dataclasses import dataclass


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class LLamaKnowledgeCreate(BaseModel):
    name: str
    size: int
    type: str
    path: str
    format: str
    os_time: int


class LlamaFileList(BaseModel):
    files: List[str]


class LlamaConfigRequest(BaseModel):
    api_key: Optional[str] = None,
    http_proxy: Optional[str] = None


@dataclass
class LlamaBaseModel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    provider: Optional[str] = None


class LlamaBaseSetting(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    apiKey: Optional[str] = None
    baseUrl: Optional[str] = None
    models: List[LlamaBaseModel] = None

@dataclass
class LLamaChatRequest(BaseModel):
    question: str = None
    # knowledge_ids: List[str] = None
    # note_ids: List[str] = None
    # access_token: str = None
    conversation_id: str = None  # 原则上这里不允许为None

@dataclass
class LlamaConversationRequest(BaseModel):
    id: str = None
    note_ids: List[str] = None
    knowledge_ids: List[str] = None
    is_pin: bool = None
    local_mode: bool = None
    create_at: float = None
    create_time: float = None
    update_at: float = None
    title: str = None
    provider_id: str = None
    model_id: str = None
    model_name: str = None
    model_path: str = None
    language_id: str = None
    system_prompt: str = ""
    language: str = None


class LLamaFileRequest(BaseModel):
    files: List[str]


class LLamaFileImportRequest(BaseModel):
    path: str


class LlamaKnowledge:
    _name: Optional[str] = None,
    _size: Optional[int] = None,
    _type: Optional[str] = None,
    _path: Optional[str] = None,
    _format: Optional[str] = None,
    _os_time: Optional[int] = None
    _file_id: Optional[str] = None,

    @property
    def file_id(self):
        return self._file_id

    @file_id.setter
    def file_id(self, value):
        self._file_id = value

    @property
    def os_time(self):
        return self._os_time

    @os_time.setter
    def os_time(self, value):
        self._os_time = value

    @property
    def format(self):
        return self._format

    @format.setter
    def format(self, value):
        self._format = value

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value
