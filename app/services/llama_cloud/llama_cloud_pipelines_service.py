import logging
import os

from dotenv import load_dotenv
from llama_cloud import PipelineCreate
from llama_cloud.client import AsyncLlamaCloud

from app.services.llama_cloud.llama_cloud_embedding_service import LlamaCloudEmbeddingService
from app.model.klee_settings import Settings as KleeSettings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LlamaCloudPipelinesService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))
        self.llama_cloud_embedding_service = LlamaCloudEmbeddingService()

    async def create_pipeline(
            self,
            embedding_config_id: str,
            name: str,
            project_id: str = None,
            organization_id: str = None
    ):
        """
        Create a pipeline
        Args:
            embedding_config_id: str
            name: str
            project_id: str
            organization_id: str
        Returns: Pipeline
        """
        request_data = {
            "embedding_config_id": embedding_config_id,
            "name": name
        }

        logger.info(f"Creating pipeline with name: {name} and embedding config id: {embedding_config_id} and project id: {project_id} and organization id: {organization_id}")

        pipeline_create_request = PipelineCreate(
            embedding_model_config_id=embedding_config_id,
            name=name,
            transform_config=None,
            data_sink_id= None,
            preset_retrieval_parameters = None,
            llama_parse_parameters = None,
            embedding_config=None,
            configured_transformations=None,
            data_sink=None,
            eval_parameters=None,
            pipeline_type=None,
            managed_pipeline_id=None
        )
        response = await self.async_client.pipelines.create_pipeline(project_id=project_id,request=pipeline_create_request)
        logger.info(f"Pipeline created with id: {response.id}")
        return response

    async def get_pipeline_by_id(
            self,
            pipeline_id: str
    ):
        """
        Get a pipeline by id
        Args:
            pipeline_id: str
        Returns: Pipeline
        """
        response = await self.async_client.pipelines.get_pipeline(pipeline_id)
        logger.info(f"Pipeline with id: {pipeline_id} retrieved")
        return response

    async def delete_pipeline_by_id(
            self,
            pipeline_id: str
    ):
        """
        Delete a pipeline by id
        Args:
            pipeline_id: str
        Returns: None
        """
        await self.async_client.pipelines.delete_pipeline(pipeline_id)
        logger.info(f"Pipeline with id: {pipeline_id} deleted")

    async def search_pipelines(
            self,
            project_id: str
    ):
        """
        Search for pipelines
        Args:
            project_id: str
        Returns: List[Pipeline]
        """
        response = await self.async_client.pipelines.search_pipelines(project_id=project_id)
        logger.info(f"Pipelines retrieved for project: {project_id}")
        logger.info(f"Search response: {response}")
        return response
