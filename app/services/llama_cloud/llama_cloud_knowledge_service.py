import logging
import os

from dotenv import load_dotenv
from llama_cloud.client import AsyncLlamaCloud
from pydantic import BaseModel

from app.config.env_config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaCloudKnowledgeRequest(BaseModel):
    project_id: str = None
    project_name: str = None

class KnowledgeCloudService:
    """
    Service class for interacting with the LlamaCloud Knowledge API.
    Used to instead of the knowledge service in the chat service.
    """
    def __init__(self):
        self.async_client = AsyncLlamaCloud(token=config.llama_cloud_api_key)

    async def create_knowledge_base(
            self,
            request: LlamaCloudKnowledgeRequest
    ):
        """
        Asynchronous function to create a knowledge base in LlamaCloud.
        Args:
            request: The request to create a knowledge base.
        Returns:
            The response from LlamaCloud.
        """
        return await self.async_client.projects.create_project(project_name=request.project_name)

    async def list_knowledge_bases(self):
        """
        Asynchronous function to list all knowledge bases in LlamaCloud.
        Returns:
            The response from LlamaCloud.
        """
        return await self.async_client.projects.list_projects()

    async def delete_knowledge_base(
            self,
            project_id: str
    ):
        """
        Asynchronous function to delete a knowledge base in LlamaCloud.
        Args:
            project_id: The ID of the knowledge base to delete.
        Returns:
            The response from LlamaCloud.
        """
        return await self.async_client.projects.delete_project(project_id=project_id)

    async def get_project(self, project_id: str):
        """
        Asynchronous function to get a project in LlamaCloud.
        Args:
            project_id: The ID of the project to get.
        Returns:
            The response from LlamaCloud.
        """
        return await self.async_client.projects.get_project(project_id=project_id)

    async def update_project(
            self,
            request: LlamaCloudKnowledgeRequest,
    ):
        """
        Asynchronous function to update a project in LlamaCloud.
        Args:
            request: The request to update a project.
        Returns:
            The response from LlamaCloud.
        """
        return await self.async_client.projects.update_project(project_id=request.project_id, project_name=request.project_name)
