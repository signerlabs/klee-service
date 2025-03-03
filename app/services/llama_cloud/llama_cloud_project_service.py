import logging
import os

from dotenv import load_dotenv
from llama_cloud import ProjectCreate
from llama_cloud.client import AsyncLlamaCloud

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaCloudProjectService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))

    async def create_project(
            self,
            request
    ):
        response = await self.async_client.projects.create_project(
            request=ProjectCreate(
                name=request.name
            )
        )
        logger.info(f"Project created: {response}")
        return response

    async def get_project(
            self,
            project_id: str
    ):
        response = await self.async_client.projects.get_project(project_id=project_id)
        logger.info(f"Project retrieved: {response}")
        return response

    async def delete_project(
            self,
            project_id: str
    ):
        response = await self.async_client.projects.delete_project(project_id=project_id)
        logger.info(f"Project deleted: {response}")
        return response

    async def list_projects(self):
        response = await self.async_client.projects.list_projects()
        logger.info(f"Projects retrieved: {response}")
        return response