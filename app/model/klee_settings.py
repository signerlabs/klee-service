import os

from dataclasses import dataclass
from typing import Optional

from llama_cloud.client import AsyncLlamaCloud
from llama_index.core.agent import AgentRunner


@dataclass
class _Settings:
    """
        Global settings
    """
    _llm_path: Optional[str] = None,
    _embed_model_path: Optional[str] = None,
    _openai_api_key: Optional[str] = None,
    _uid: Optional[str] = None,
    _os_type: Optional[str] = None,
    _vector_url: Optional[str] = None,
    _config_url: Optional[str] = None,
    _data: Optional[dict] = None,
    _temp_file_url: Optional[str] = None,
    _chat_engine_dict: Optional[dict] = None
    _checked_chat_engine: Optional[AgentRunner] = None
    _agent_dict: Optional[dict] = None
    _dictionary: Optional[dict] = None
    _local_mode: Optional[bool] = True
    _center_url: Optional[str] = None
    _un_load: Optional[bool] = True
    _global_model_id: Optional[str] = None
    _global_model_path: Optional[str] = None
    _global_model_name: Optional[str] = None
    _provider_id: Optional[str] = None
    _agent_runner: Optional[AgentRunner] = None
    _async_llama_cloud: Optional[AsyncLlamaCloud] = None

    @property
    def async_llama_cloud(self):
        return self._async_llama_cloud

    @async_llama_cloud.setter
    def async_llama_cloud(self, value):
        self._async_llama_cloud = value

    @property
    def agent_runner(self):
        return self._agent_runner

    @agent_runner.setter
    def agent_runner(self, value):
        self._agent_runner = value

    @property
    def provider_id(self):
        return self._provider_id

    @provider_id.setter
    def provider_id(self, value):
        self._provider_id = value

    @property
    def global_model_name(self):
        return self._global_model_name

    @global_model_name.setter
    def global_model_name(self, value):
        self._global_model_name = value

    @property
    def global_model_path(self):
        return self._global_model_path

    @global_model_path.setter
    def global_model_path(self, value):
        self._global_model_path = value

    @property
    def global_model_id(self):
        return self._global_model_id

    @global_model_id.setter
    def global_model_id(self, value):
        self._global_model_id = value

    @property
    def un_load(self):
        return self._un_load

    @un_load.setter
    def un_load(self, value):
        self._un_load = value

    @property
    def center_url(self):
        return self._center_url

    @center_url.setter
    def center_url(self, value):
        self._center_url = value

    @property
    def local_mode(self):
        return self._local_mode

    @local_mode.setter
    def local_mode(self, value):
        self._local_mode = value

    @property
    def dictionary(self):
        return self._dictionary

    @dictionary.setter
    def dictionary(self, value):
        self._dictionary = value

    @property
    def agent_dict(self):
        return self._agent_dict

    @agent_dict.setter
    def agent_dict(self, agent_dict):
        """
            这里碍于入参设置，因此需要把整个字典传进来覆盖掉
        """
        self._agent_dict = agent_dict

    @property
    def checked_chat_engine(self):
        return self._checked_chat_engine

    @checked_chat_engine.setter
    def checked_chat_engine(self, value):
        self._checked_chat_engine = value

    @property
    def chat_engine_dict(self):
        return self._chat_engine_dict

    @chat_engine_dict.setter
    def chat_engine_dict(self, value):
        self._chat_engine_dict = value

    @property
    def temp_file_url(self) -> str:
        return self._temp_file_url

    @temp_file_url.setter
    def temp_file_url(self, value: str):
        self._temp_file_url = value

    @property
    def llm_path(self) -> str:
        return self._llm_path

    @llm_path.setter
    def llm_path(self, llm_path: str):
        self._llm_path = llm_path

    @property
    def embed_model_path(self):
        return self._embed_model_path

    @embed_model_path.setter
    def embed_model_path(self, value: str):
        self._embed_model_path = value

    @property
    def config_url(self) -> str:
        return self._config_url

    @config_url.setter
    def config_url(self, config_url: str):
        self._config_url = config_url

    @property
    def vector_url(self) -> str:
        return self._vector_url

    @vector_url.setter
    def vector_url(self, vector_url: str):
        self._vector_url = vector_url

    @property
    def os_type(self) -> str:
        if self._os_type is None:
            if os.name == 'nt':
                self._os_type = "Windows"
            elif os.name == "posix":
                self._os_type = "Linux/macOS"
        return self._os_type

    @os_type.setter
    def os_type(self, value):
        self._os_type = value

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def openai_api_key(self) -> str:
        return self._openai_api_key

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: Optional[dict]):
        self._data = data


Settings = _Settings()
