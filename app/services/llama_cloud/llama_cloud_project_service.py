import logging

from llama_cloud import ProjectCreate
from app.model.klee_settings import Settings as KleeSettings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LlamaCloudProjectService:
    def __init__(self):
        logger.info("Initializing LlamaCloudProjectService")
        self.async_client = KleeSettings.async_llama_cloud

    async def create_project(
            self,
            request
    ):
        """
        Create a new project in LlamaCloud
        Args:
            request: ProjectCreate
        Returns:
            Project
        """
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
        """
        Get a project by ID in LlamaCloud
        Args:
            project_id: str
        Returns:
            Project
        """
        response = await self.async_client.projects.get_project(project_id=project_id)
        logger.info(f"Project retrieved: {response}")
        return response

    async def delete_project(
            self,
            project_id: str
    ):
        """
        Delete a project by ID in LlamaCloud
        Args:
            project_id: str
        Returns:
            None
        """
        response = await self.async_client.projects.delete_project(project_id=project_id)
        logger.info(f"Project deleted: {response}")
        return response

    async def list_projects(self):
        response = await self.async_client.projects.list_projects()
        logger.info(f"Projects retrieved: {response}")
        return response