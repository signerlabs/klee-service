import logging
import os
import typing

from dotenv import load_dotenv
from llama_cloud import EmbeddingModelConfigCreateEmbeddingConfig, EmbeddingModelConfig
from llama_cloud.client import AsyncLlamaCloud
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LlamaCloudEmbeddingConfigRequest(BaseModel):
    embedding_config_id: str = None
    project_id: str = None
    name: str = None


class LlamaCloudEmbeddingService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))
        self.openai_api_key = os.getenv("OPENAI_KEY")
        logger.info(f"Llama Cloud API Key: {os.getenv('LLAMA_CLOUD_API_KEY')}")
        logger.info(f"OpenAI API Key: {self.openai_api_key}")

    async def create_embedding_config(
            self,
            request: LlamaCloudEmbeddingConfigRequest
    ):
        logger.info(f"Create embedding config request: {request}")
        logger.info(f"OpenAI API Key: {self.openai_api_key}")
        embedding_config = {
            "type": "OPENAI_EMBEDDING",
            "component": {
                "model_name": "text-embedding-ada-002",
                "embed_batch_size": 10,
                "num_workers": 1,
                "api_key": self.openai_api_key,
                "max_retries": 5,
                "timeout": 60
            }
        }

        transformer_config = {
            "mode": "auto",
            "chunk_size": 1024,
            "chunk_overlap": 200,
        }

        response_embedding_config = await self.async_client.embedding_model_configs.create_embedding_model_config(
            project_id=request.project_id,
            name=request.name,
            embedding_config=embedding_config,
            transformer_config=transformer_config
        )
        logger.info(f"Create embedding config response: {response_embedding_config}")
        return None

    async def update_embedding_config(
            self,
            request: LlamaCloudEmbeddingConfigRequest
    ):
        response_embedding_config = await self.async_client.embedding_model_configs.update_embedding_model_config(
            embedding_model_config_id=request.embedding_config_id,
            embedding_config=EmbeddingModelConfigCreateEmbeddingConfig(
                model_name="text-embedding-ada-002",
                embed_batch_size=10,
                num_workers=1,
                api_key=os.getenv("OPENAI_API_KEY"),
                max_retries=5,
                timeout=60
            )
        )
        logger.info(f"Update embedding config response: {response_embedding_config}")
        return None

    async def delete_embedding_config(
            self,
            request: LlamaCloudEmbeddingConfigRequest
    ):
        response_embedding_config = await self.async_client.embedding_model_configs.delete_embedding_model_config(
            embedding_model_config_id=request.embedding_config_id
        )
        logger.info(f"Delete embedding config response: {response_embedding_config}")
        return None

    async def list_embedding_configs(
            self,
            project_id: str
    ) -> typing.List[EmbeddingModelConfig]:
        """
        List all embedding configs for a given project.
        Args:
            project_id: The project id to list embedding configs for.
        Returns:
            A list of embedding configs for the given project.
        """
        response_embedding_configs = await self.async_client.embedding_model_configs.list_embedding_model_configs(
            project_id=project_id
        )
        logger.info(f"List embedding configs response: {response_embedding_configs}")
        return response_embedding_configs

    async def get_embedding_config(
            self,
            embedding_config_id: str,
            project_id: str
    ) -> EmbeddingModelConfig | None:
        response = await self.list_embedding_configs(project_id="default")
        for embedding_config in response:
            if embedding_config.id == embedding_config_id:
                return embedding_config
        return None