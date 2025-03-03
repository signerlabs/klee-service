import datetime
import json
import logging
import os
import uuid

import requests
from sqlalchemy import select, delete
from llama_index.core.settings import Settings

from app.common.LlamaEnum import SystemTypeDiffModelType
from app.model.LlamaRequest import LlamaBaseSetting, LlamaConversationRequest
from app.model.Response import ResponseContent
from app.model.base_config import BaseConfig
from app.model.chat_message import Conversation
from app.model.global_settings import GlobalSettings
from app.model.knowledge import File
from app.services.client_sqlite_service import db_transaction
from app.services.llama_index_service import LlamaIndexService
from app.model.klee_settings import Settings as KleeSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseService:
    def __init__(self):
        logger.info("BaseService initialized")
        self.llama_index_service = LlamaIndexService()

    @db_transaction
    async def create_providers(
            self,
            base_request: LlamaBaseSetting,
            session = None
    ):
        try:
            now_time = datetime.datetime.now().timestamp()
            config_id = str(uuid.uuid4())

            model_dict_arr = [
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "icon": item.icon,
                    "provider": item.provider
                }
                for item in base_request.models
            ]

            base_config = BaseConfig(
                id=config_id,
                apiKey=base_request.apiKey,
                description=base_request.description,
                name=base_request.name,
                baseUrl=base_request.baseUrl,
                models=json.dumps(model_dict_arr),
                create_at=now_time,
                update_at=now_time,
                delete_at=0
            )

            session.add(base_config)
            base_request.id = config_id
            return ResponseContent(error_code=0, message="Add model successfully", data=base_request)
        except Exception as e:
            logger.error(f"create_providers error: {e}")
            return ResponseContent(error_code=1, message="Add model failed,", data=None)

    @db_transaction
    async def update_providers(
            self,
            provider_id: str,
            base_request: LlamaBaseSetting,
            session = None
    ):
        try:
            stmt = select(BaseConfig).where(BaseConfig.id == provider_id)
            base_result = await session.execute(stmt)
            base_config = base_result.scalars().one_or_none()

            now_time = datetime.datetime.now().timestamp()

            model_dict_arr = [
                {
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "icon": item.icon,
                    "provider": item.provider
                }
                for item in base_request.models
            ]

            base_config.apiKey = base_request.apiKey
            base_config.description = base_request.description
            base_config.name = base_request.name
            base_config.baseUrl = base_request.baseUrl
            base_config.models = json.dumps(model_dict_arr)
            now_time = datetime.datetime.now().timestamp()
            base_config.update_at = now_time
            base_request.id = base_config.id

            session.add(base_config)
            return ResponseContent(error_code=0, message="Update model successfully", data=base_request)
        except Exception as e:
            logger.error(f"update_providers error: {e}")
            return ResponseContent(error_code=-1, message=f"Update model failed, {str(e)}", data={})

    @db_transaction
    async def delete_providers(
            self,
            provider_id: str,
            session = None
    ):
        try:
            delete_stmt = delete(BaseConfig).where(BaseConfig.id == provider_id)
            await session.execute(delete_stmt)

            return ResponseContent(error_code=0, message=f"Delete model successfully", data={})
        except Exception as e:
            logger.error(f"update_providers error: {e}")
            return ResponseContent(error_code=-1, message=f"Delete model failed，{str(e)}", data={})

    @db_transaction
    async def get_all_providers(
            self,
            session
    ):
        try:
            stmt = select(BaseConfig)
            results = await session.execute(stmt)
            provider_results = results.scalars().all()
            providers = [
                {
                    "id": provider.id,
                    "name": provider.name,
                    "description": provider.description,
                    "apiKey": provider.apiKey,
                    "baseUrl": provider.baseUrl,
                    "models": json.loads(provider.models)
                }
                for provider in provider_results
            ]

            return ResponseContent(error_code=0, message="Get model list successfully", data=providers)
        except Exception as e:
            logger.error(f"get_all_providers error: {e}")
            return ResponseContent(error_code=-1, message=f"Get model list failed, {str(e)}", data=[])

    @db_transaction
    async def update_conversation_setting(
            self,
            llama_request: LlamaConversationRequest,
            session
    ):
        try:
            stmt = select(Conversation).where(Conversation.id == llama_request.id)
            result = await session.execute(stmt)
            conversation = result.scalars().first()

            if llama_request.provider_id == SystemTypeDiffModelType.KLEE.value and \
                    not os.path.exists(f"{KleeSettings.llm_path}{str(llama_request.model_id).lower()}.gguf"):
                return ResponseContent(error_code=-1, message="unexist model", data={})

            if llama_request.provider_id == SystemTypeDiffModelType.OLLAMA.value:
                try:
                    with requests.get("http://localhost:11434/api/tags") as response:
                        response.raise_for_status()
                        response.encoding = "utf-8"
                except Exception as e:
                    logging.error(f"Ollama not installed or not started, {str(e)}")
                    return ResponseContent(error_code=-1, message=f"Ollama not installed or not started", data={})

            if conversation.model_id != llama_request.model_id:
                self.llama_index_service.release_memory()
            elif llama_request.model_path != conversation.model_path:
                self.llama_index_service.release_memory()

            KleeSettings.local_mode = llama_request.local_mode

            conversation.knowledge_ids = json.dumps(llama_request.knowledge_ids, ensure_ascii=False)
            conversation.note_ids = json.dumps(llama_request.note_ids, ensure_ascii=False)
            conversation.local_mode = llama_request.local_mode
            conversation.language_id = llama_request.language_id
            conversation.provider_id = llama_request.provider_id
            conversation.model_id = llama_request.model_id
            conversation.language_id = llama_request.language_id
            conversation.system_prompt = llama_request.system_prompt
            conversation.model_name = llama_request.model_name
            conversation.model_path = llama_request.model_path
            provider_id = llama_request.provider_id
            model_id = llama_request.model_id
            model_path = llama_request.model_path
            model_name = llama_request.model_name

            stmt_global = select(GlobalSettings)
            result_global = await session.execute(stmt_global)
            global_settings = result_global.scalars().first()

            print(
                f"global settings info: {global_settings.provider_id}, {global_settings.model_id}, {global_settings.model_path}, {global_settings.model_name}, {global_settings.local_mode}")

            if conversation.provider_id == SystemTypeDiffModelType.OLLAMA.value and conversation.model_id != global_settings.model_id:
                if global_settings.provider_id == SystemTypeDiffModelType.OLLAMA.value:
                    try:
                        os.system(f"ollama stop {global_settings.model_id}")
                    except Exception as e:
                        logging.error(f"ollama stop {global_settings.model_id} failed, {str(e)}")
                        pass

            if global_settings is not None:
                if global_settings.provider_id != provider_id \
                        or global_settings.model_id != model_id \
                        or global_settings.model_path != model_path \
                        or global_settings.model_name != model_name:
                    global_settings.provider_id = provider_id
                    global_settings.model_id = model_id
                    global_settings.model_path = model_path
                    global_settings.model_name = model_name
                    global_settings.local_mode = llama_request.local_mode

                    KleeSettings.local_mode = llama_request.local_mode
                    KleeSettings.provider_id = provider_id
                    KleeSettings.model_id = model_id
                    KleeSettings.model_path = model_path
                    KleeSettings.model_name = model_name

                    KleeSettings.un_load = True

                    with self.llama_index_service.release_memory():
                        Settings.llm = None
                        self.llama_index_service.release_memory()

            session.add(conversation)
            await session.flush()

            file_infos = {}
            knowledge_ids = json.loads(conversation.knowledge_ids)
            if len(knowledge_ids) > 0:
                for knowledge_id in knowledge_ids:
                    stmt = select(File).where(File.knowledgeId == knowledge_id)
                    result = await session.execute(stmt)
                    files = result.scalars().all()

                    if len(files) > 0:
                        file_infos[knowledge_id] = files

            response_data = {
                "id": conversation.id,
                "knowledge_ids": conversation.knowledge_ids,
                "note_ids": conversation.note_ids,
                "local_mode": conversation.local_mode,
                "provider_id": conversation.provider_id,
                # "is_pin": conversation.is_pin,
                "model_id": conversation.model_id,
                "model_name": conversation.model_name,
                "language_id": conversation.language_id,
                "system_prompt": conversation.system_prompt,
                "model_path": conversation.model_path
            }
            return ResponseContent(error_code=0, message="update successfully", data=response_data)
        except Exception as e:
            # 比较笨
            logger.error(f"update_conversation_setting error: {e}")
            return ResponseContent(error_code=-1, message=f"updated failed, {str(e)}", data={})

    async def get_status(self):
        return ResponseContent(error_code=0, message="service is running", data={})

