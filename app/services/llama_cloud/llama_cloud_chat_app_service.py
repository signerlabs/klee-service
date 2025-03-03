import logging
import os
import typing

from dotenv import load_dotenv
from llama_cloud import PresetCompositeRetrievalParams, CompositeRetrievalMode, LlmParameters, SupportedLlmModelNames
from llama_cloud.client import AsyncLlamaCloud

from app.services.llama_cloud.llama_cloud_retrievers_service import LlamaCloudRetrieversService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LlamaCloudChatAppService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))
        self.retrievers_service = LlamaCloudRetrieversService()

    async def create_chat_app_func(
            self,
            retriever_id: str,
            name: str
    ):
        await self.async_client.chat_apps.create_chat_app_api_v_1_apps_post(
            name=name,
            retriever_id=retriever_id,
            llm_config=LlmParameters(
                system_prompt="You are a helpful assistant.",
                temperature=0.5,
                model_name=SupportedLlmModelNames.GPT_4_O,
                use_chain_of_thought_reasoning=True,
                use_citation=True,
                class_name="base_component"
            ),
            retrieval_config = PresetCompositeRetrievalParams(
                mode=CompositeRetrievalMode.FULL,
                rerank_top_n=6
            )
        )
        logger.info(f"Created chat app with name {name} and retriever_id {retriever_id}")

    async def get_chat_app_func(
            self,
            chat_app_id: str
    ):
        chat_app = await self.async_client.chat_apps.get_chat_app_api_v_1_apps_chat_app_id_get(chat_app_id)
        logger.info(f"Got chat app with id {chat_app_id}: {chat_app}")
        return chat_app

    async def chat_with_chat_app(
            self,
            chat_app_id: str,
            messages: typing.List[str]
    ):
        response = await self.async_client.chat_apps.chat_with_chat_app(chat_app_id, messages)
        logger.info(f"Got response from chat app with id {chat_app_id}: {response}")
        return response

    async def get_chat_apps(
            self
    ):
        chat_apps = await self.async_client.chat_apps.get_chat_apps_api_v_1_apps_get()
        logger.info(f"Got chat apps: {chat_apps}")
        return chat_apps
