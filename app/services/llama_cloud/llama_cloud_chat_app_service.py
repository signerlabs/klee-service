import logging
import os
import typing

import httpx
from dotenv import load_dotenv
from httpx import Timeout
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
            retrieval_config=PresetCompositeRetrievalParams(
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
        messages = [
            {
                "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "role": "user",
                "content": "介绍一下蔡立堉",
                "data": {},
                "class_name": "base_component"
            }
        ]
        response = await self.async_client.chat_apps.chat_with_chat_app(id=chat_app_id, messages=messages)
        logger.info(f"Got response from chat app with id {chat_app_id}: {response}")
        return response

    async def get_chat_apps(
            self
    ):
        chat_apps = await self.async_client.chat_apps.get_chat_apps_api_v_1_apps_get()
        logger.info(f"Got chat apps: {chat_apps}")
        return chat_apps

    async def rpc_get_chat_apps(self):
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('LLAMA_CLOUD_API_KEY')}"
            }
            response = await client.get("https://api.cloud.llamaindex.ai/api/v1/apps/", headers=headers)
            response.raise_for_status()
            logger.info(f"Got chat apps: {response.json()}")
            return response.json()

    async def rpc_create_chat_app(self, retriever_id: str, name: str):
        async with httpx.AsyncClient() as client:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('LLAMA_CLOUD_API_KEY')}"
            }
            response = await client.post(
                "https://api.cloud.llamaindex.ai/api/v1/apps/",
                headers=headers,
                json={
                    "retriever_id": retriever_id,
                    "name": "my test chat app",
                    "llm_config": {
                        "system_prompt": "You are a helpful assistant.",
                        "temperature": 0.5,
                        "model_name": "GPT_4O",
                        "use_chain_of_thought_reasoning": True,
                        "use_citation": True,
                        "class_name": "base_component"
                    },
                    "retrieval_config": {
                        "mode": "full",
                        "rerank_top_n": 6
                    }
                }
            )
            response.raise_for_status()
            logger.info(f"Created chat app with name {name} and retriever_id {retriever_id}")
            logger.info(f"Got response: {response.json()}")
            return response.json()

    async def rpc_chat_with_chat_app(
            self,
            chat_app_id: str,
            messages,
    ):
        async with httpx.AsyncClient(timeout=Timeout(timeout=120.0)) as client:
            logger.info(f"Sending messages to chat app with id {chat_app_id}: {messages}")
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {os.getenv('LLAMA_CLOUD_API_KEY')}"
            }
            response = await client.post(
                f"https://api.cloud.llamaindex.ai/api/v1/apps/{chat_app_id}/chat",
                headers=headers,
                json={
                    "messages": [
                        {
                            "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                            "role": "user",
                            "content": "分别介绍一下蔡立堉和林伟光",
                            "data": {},
                            "class_name": "base_component"
                        }
                    ]
                }
            )
            response.raise_for_status()

            # 解码字节数据为字符串
            decoded_data = response.content.decode('utf-8')

            # 按行分割
            lines = decoded_data.splitlines()

            # 提取每行的有效部分并拼接
            result = ''
            for line in lines:
                if line.startswith('0:"'):
                    # 提取引号内的内容
                    content = line.split('"')[1]
                    result += content

            return result
