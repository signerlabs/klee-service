import logging
import os

from dotenv import load_dotenv
from llama_cloud import RetrieverCreate
from llama_cloud.client import AsyncLlamaCloud

from app.services.llama_cloud.llama_cloud_pipelines_service import LlamaCloudPipelinesService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaCloudRetrieversService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))
        self.pipelines_service = LlamaCloudPipelinesService()

    async def create_retriever(
            self,
            retriever_name: str,
            project_id: str
    ):
        pipelines = await self.pipelines_service.search_pipelines(project_id=project_id)
        await self.async_client.retrievers.create_retriever(
            project_id=project_id,
            request=RetrieverCreate(
                name=retriever_name,
                pipelines=pipelines
            )
        )
        logger.info(f"Retriever {retriever_name} created successfully")

    async def get_retriever(
            self,
            retriever_id: str
    ):
        response = await self.async_client.retrievers.get_retriever(retriever_id=retriever_id)
        logger.info(f"Retriever {retriever_id} retrieved successfully")
        return response

    async def delete_retriever(
            self,
            retriever_id: str
    ):
        await self.async_client.retrievers.delete_retriever(retriever_id=retriever_id)
        logger.info(f"Retriever {retriever_id} deleted successfully")

    async def retrieve(
            self,
            retriever_id: str,
            query: str
    ):
        response = await self.async_client.retrievers.retrieve(
            retriever_id=retriever_id,
            query=query
        )
        logger.info(f"Retriever {retriever_id} retrieved successfully")
        return response

    async def list_retrievers(
            self
    ):
        response = await self.async_client.retrievers.list_retrievers()
        logger.info(f"Retrievers retrieved successfully")
        return response