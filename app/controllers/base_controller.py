import logging

from fastapi import APIRouter, Depends

# 主要用于配置对话的基本设置
from app.model.LlamaRequest import LlamaConversationRequest
from app.model.LlamaRequest import LlamaBaseSetting

from app.services.base_service import BaseService

router = APIRouter()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

class BaseController:
    def __init__(self):
        self.base_service = BaseService()

    async def create_providers(
            self,
            base_request: LlamaBaseSetting,
    ):
        return await self.base_service.create_providers(base_request)

    async def update_providers(
            self,
            provider_id: str,
            base_request: LlamaBaseSetting
    ):
        return await self.base_service.update_providers(provider_id, base_request)

    async def delete_providers(
            self,
            provider_id: str
    ):
        return await self.base_service.delete_providers(provider_id)

    async def get_all_providers(self):
        return await self.base_service.get_all_providers()

    async def get_status(self):
        return await self.base_service.get_status()

    async def update_conversation_setting(
            self,
            llama_request: LlamaConversationRequest
    ):
        return await self.base_service.update_conversation_setting(llama_request)

@router.post("/custom_providers")
async def create_providers(
        base_request: LlamaBaseSetting,
        controller: BaseController = Depends(BaseController)
):
    """
        自定义提供商的llm方法
    """
    return await controller.create_providers(base_request)

@router.put("/custom_providers/{provider_id}")
async def update_providers(
        provider_id: str,
        base_request: LlamaBaseSetting,
        controller: BaseController = Depends(BaseController)
):
    return await controller.update_providers(provider_id, base_request)


@router.delete("/custom_providers/{provider_id}")
async def delete_providers(
        provider_id: str,
        controller: BaseController = Depends(BaseController)
):
    return await controller.delete_providers(provider_id)


@router.get("/custom_providers")
async def get_all_providers(
        controller: BaseController = Depends(BaseController)
):
    return await controller.get_all_providers()

@router.get("/status")
async def status(
        controller: BaseController = Depends(BaseController)
):
    """
    Check health status
    """
    return await controller.get_status()

@router.put("/conversation/setting")
async def update_conversation_setting(
        llama_request: LlamaConversationRequest,
        controller: BaseController = Depends(BaseController)
):
    return await controller.update_conversation_setting(llama_request)
