from typing import Dict, Any
import datetime
import json
import logging
import os
import uuid
from enum import Enum

import requests
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ErrorCode(Enum):
    SUCCESS = 0
    GENERAL_ERROR = -1
    MODEL_NOT_FOUND = -2

class BaseService:
    def __init__(self):
        logger.info("BaseService initialized")
        self.llama_index_service = LlamaIndexService()

    def _create_response(
        self, 
        error_code: ErrorCode, 
        message: str, 
        data: Any = None
    ) -> ResponseContent:
        """创建统一的响应对象"""
        return ResponseContent(
            error_code=error_code.value,
            message=message,
            data=data
        )

    def _model_to_dict(self, model: Any) -> Dict:
        """将模型对象转换为字典"""
        return {
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "icon": model.icon,
            "provider": model.provider
        }

    @db_transaction
    async def create_providers(
            self,
            base_request: LlamaBaseSetting,
            session: AsyncSession
    ) -> ResponseContent:
        try:
            now_time = datetime.datetime.now().timestamp()
            config_id = str(uuid.uuid4())

            model_dict_arr = [self._model_to_dict(item) for item in base_request.models]

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
            return self._create_response(
                ErrorCode.SUCCESS,
                "Add model successfully",
                base_request
            )
        except Exception as e:
            logger.error(f"create_providers error: {str(e)}", exc_info=True)
            return self._create_response(
                ErrorCode.GENERAL_ERROR,
                f"Failed to add model: {str(e)}"
            )

    @db_transaction
    async def update_providers(
            self,
            provider_id: str,
            base_request: LlamaBaseSetting,
            session: AsyncSession
    ) -> ResponseContent:
        try:
            stmt = select(BaseConfig).where(BaseConfig.id == provider_id)
            base_result = await session.execute(stmt)
            base_config = base_result.scalars().one_or_none()

            if not base_config:
                return self._create_response(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Provider with id {provider_id} not found"
                )

            model_dict_arr = [self._model_to_dict(item) for item in base_request.models]
            now_time = datetime.datetime.now().timestamp()

            # 更新配置
            base_config.apiKey = base_request.apiKey
            base_config.description = base_request.description
            base_config.name = base_request.name
            base_config.baseUrl = base_request.baseUrl
            base_config.models = json.dumps(model_dict_arr)
            base_config.update_at = now_time
            
            base_request.id = base_config.id
            session.add(base_config)

            return self._create_response(
                ErrorCode.SUCCESS,
                "Update model successfully",
                base_request
            )
        except Exception as e:
            logger.error(f"update_providers error: {str(e)}", exc_info=True)
            return self._create_response(
                ErrorCode.GENERAL_ERROR,
                f"Failed to update model: {str(e)}"
            )

    @db_transaction
    async def delete_providers(
            self,
            provider_id: str,
            session: AsyncSession
    ) -> ResponseContent:
        try:
            # 先检查是否存在
            stmt = select(BaseConfig).where(BaseConfig.id == provider_id)
            result = await session.execute(stmt)
            if not result.scalars().first():
                return self._create_response(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Provider with id {provider_id} not found"
                )

            delete_stmt = delete(BaseConfig).where(BaseConfig.id == provider_id)
            await session.execute(delete_stmt)

            return self._create_response(
                ErrorCode.SUCCESS,
                "Delete model successfully"
            )
        except Exception as e:
            logger.error(f"delete_providers error: {str(e)}", exc_info=True)
            return self._create_response(
                ErrorCode.GENERAL_ERROR,
                f"Failed to delete model: {str(e)}"
            )

    @db_transaction
    async def get_all_providers(
            self,
            session: AsyncSession
    ) -> ResponseContent:
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

            return self._create_response(
                ErrorCode.SUCCESS,
                "Get model list successfully",
                providers
            )
        except Exception as e:
            logger.error(f"get_all_providers error: {str(e)}", exc_info=True)
            return self._create_response(
                ErrorCode.GENERAL_ERROR,
                f"Failed to get model list: {str(e)}"
            )

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
            return ResponseContent(error_code=0, message="Update successful", data=response_data)
        except Exception as e:
            logger.error(f"update_conversation_setting error: {e}")
            return ResponseContent(error_code=-1, message=f"Update failed: {str(e)}", data={})

    async def get_status(self):
        return ResponseContent(error_code=0, message="Service is running", data={})
