import logging
from typing import List

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.services.llama_cloud.llama_cloud_chat_app_service import LlamaCloudChatAppService
from app.services.llama_cloud.llama_cloud_chat_service import ChatCloudService
from app.services.llama_cloud.llama_cloud_data_sink_service import LlamaCloudDataSinkService
from app.services.llama_cloud.llama_cloud_embedding_service import LlamaCloudEmbeddingService, \
    LlamaCloudEmbeddingConfigRequest
from app.services.llama_cloud.llama_cloud_file_service import LlamaCloudFileService
from app.services.llama_cloud.llama_cloud_pipelines_service import LlamaCloudPipelinesService
from app.services.llama_cloud.llama_cloud_project_service import LlamaCloudProjectService
from app.services.llama_cloud.llama_cloud_retrievers_service import LlamaCloudRetrieversService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()


class LlamaCloudProjectRequest(BaseModel):
    name: str


class LlamaCloudPipelineRequest(BaseModel):
    name: str
    embedding_config_id: str
    project_id: str = None

class LlamaCloudRetrieverRequest(BaseModel):
    project_id: str = None
    retriever_name: str

class LlamaCloudChatAppRequest(BaseModel):
    retriever_id: str
    name: str

class LlamaCloudController:
    def __init__(self):
        self.chat_cloud_service = ChatCloudService()
        self.file_service = LlamaCloudFileService()
        self.project_service = LlamaCloudProjectService()
        self.embeddings_service = LlamaCloudEmbeddingService()

    async def achat(
            self,
            message: str,
            knowledge_base_ids: List[str],
    ):
        """
        Asynchronous function to chat with LlamaCloud.
        Args:
            message: The message to send to LlamaCloud.
            knowledge_base_ids: The knowledge base ids to use for the chat.

        Returns:
            The response from LlamaCloud.
        """
        response = await self.chat_cloud_service.achat(message, knowledge_base_ids)
        logger.info(f"LlamaCloud response: {response}")
        return response

    async def a_upload_file(
            self,
            file_path: str
    ):
        """
        Asynchronous function to upload a file to LlamaCloud.
        Args:
            file_path: The path of the file to upload.

        Returns:
            The response from LlamaCloud.
        """
        response = await self.file_service.upload_file(file_path)
        logger.info(f"LlamaCloud file upload response: {response}")
        return response

    async def list_files(self):
        """
        Asynchronous function to list all files in LlamaCloud.
        Returns:
            The response from LlamaCloud.
        """
        response = await self.file_service.get_files()
        logger.info(f"LlamaCloud list files response: {response}")
        return response

    async def get_file_content(self, file_id: str):
        """
        Asynchronous function to get the content of a file in LlamaCloud.
        Args:
            file_id: The id of the file.
        Returns:
            The content of the file.
        """
        return await self.file_service.read_file_content(file_id)

    async def create_project(
            self,
            request: LlamaCloudProjectRequest,
    ):
        """
        Asynchronous function to create a project in LlamaCloud.
        Args:
            request: The request body containing the project name.
        Returns:
            The response from LlamaCloud.
        """
        return await self.project_service.create_project(request=request)

    async def list_projects(self):
        """
        Asynchronous function to list all projects in LlamaCloud.
        Returns:
            The response from LlamaCloud.
        """
        return await self.project_service.list_projects()


class LlamaCloudChatRequest(BaseModel):
    message: str
    knowledge_base_ids: List[str]


class LlamaCloudFileUploadRequest(BaseModel):
    file_path: str


@router.post("/achat")
async def chat_with_llama_cloud(
        request: LlamaCloudChatRequest,
        controller: LlamaCloudController = Depends(LlamaCloudController)
):
    """
    Endpoint to chat with LlamaCloud.
    Args:
        request: The request body containing the message and knowledge base ids.
        controller: LlamaCloudController = Depends(LlamaCloudController)
    Returns:
        The response from LlamaCloud.
    """
    response = await controller.achat(request.message, request.knowledge_base_ids)
    return response


@router.post("/cloud/upload_file")
async def upload_file_to_llama_cloud(
        request: LlamaCloudFileUploadRequest,
        controller: LlamaCloudController = Depends(LlamaCloudController)
):
    """
    Endpoint to upload a file to LlamaCloud.
    Args:
        request: The request body containing the file path.
        controller: LlamaCloudController = Depends(LlamaCloudController)
    Returns:
        The response from LlamaCloud.
    """
    return await controller.a_upload_file(request.file_path)


@router.get("/list_files")
async def list_files_in_llama_cloud(
        controller: LlamaCloudController = Depends(LlamaCloudController)
):
    """
    Endpoint to list all files in LlamaCloud.
    Args:
        controller: LlamaCloudController = Depends(LlamaCloudController)
    Returns:
        The response from LlamaCloud.
    """
    return await controller.list_files()


@router.get("/{file_id}/content")
async def get_file_content(
        file_id: str,
        controller: LlamaCloudController = Depends(LlamaCloudController)
):
    """
    Endpoint to get the content of a file in LlamaCloud.
    Args:
        file_id: The id of the file.
        controller: LlamaCloudController = Depends(LlamaCloudController)
    Returns:
        The content of the file.
    """
    return await controller.get_file_content(file_id)


@router.post("/create_project")
async def create_project_in_llama_cloud(
        request: LlamaCloudProjectRequest,
        controller: LlamaCloudController = Depends(LlamaCloudController)
):
    """
    Endpoint to create a project in LlamaCloud.
    Args:
        request: The request body containing the project name.
        controller: LlamaCloudController = Depends(LlamaCloudController)
    Returns:
        The response from LlamaCloud.
    """
    return await controller.create_project(request)


@router.get("/list_projects/test")
async def list_projects_in_llama_cloud(
        controller: LlamaCloudController = Depends(LlamaCloudController)
):
    """
    Endpoint to list all projects in LlamaCloud.
    Args:
        controller: LlamaCloudController = Depends(LlamaCloudController)
    Returns:
        The response from LlamaCloud.
    """
    logger.info("list_projects_in_llama_cloud")
    return await controller.list_projects()


@router.post("/create_embedding_config")
async def create_embedding_config(
        request: LlamaCloudEmbeddingConfigRequest,
        service: LlamaCloudEmbeddingService = Depends(LlamaCloudEmbeddingService)
):
    """
    Endpoint to create an embedding config in LlamaCloud.
    Args:
        request: The request body containing the embedding config.
        service: LlamaCloudEmbeddingService = Depends(LlamaCloudEmbeddingService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.create_embedding_config(request)


@router.get("/list_embedding_configs/{project_id}")
async def list_embedding_configs(
        project_id: str,
        service: LlamaCloudEmbeddingService = Depends(LlamaCloudEmbeddingService)
):
    """
    Endpoint to list all embedding configs in LlamaCloud.
    Args:
        project_id: The id of the project.
        service: LlamaCloudEmbeddingService = Depends(LlamaCloudEmbeddingService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.list_embedding_configs(project_id=project_id)


@router.post("/create_pipeline")
async def create_pipeline(
        request: LlamaCloudPipelineRequest,
        service: LlamaCloudPipelinesService = Depends(LlamaCloudPipelinesService)
):
    """
    Endpoint to create a pipeline in LlamaCloud.
    Args:
        request: The request body containing the pipeline name, embedding config id, and project id.
        service: LlamaCloudPipelinesService = Depends(LlamaCloudPipelinesService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.create_pipeline(
        project_id=request.project_id,
        embedding_config_id=request.embedding_config_id,
        name=request.name
    )


@router.get("/search_pipelines/{project_id}")
async def search_pipelines(
        project_id: str,
        service: LlamaCloudPipelinesService = Depends(LlamaCloudPipelinesService)
):
    """
    Endpoint to search for pipelines in LlamaCloud.
    Args:
        project_id: The id of the project.
        service: LlamaCloudPipelinesService = Depends(LlamaCloudPipelinesService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.search_pipelines(project_id=project_id)


@router.get("/list_data_sinks")
async def list_data_sinks(
        service: LlamaCloudDataSinkService = Depends(LlamaCloudDataSinkService)
):
    """ Endpoint to list all data sinks in LlamaCloud.
    Args:
            service: LlamaCloudDataSinkService = Depends(LlamaCloudDataSinkService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.list_data_sinks()

@router.delete("/projects/{project_id}")
async def delete_project(
        project_id: str,
        service: LlamaCloudProjectService = Depends(LlamaCloudProjectService)
):
    """ Endpoint to delete a project in LlamaCloud.
    Args:
            project_id: The id of the project.
            service: LlamaCloudProjectService = Depends(LlamaCloudProjectService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.delete_project(project_id)

@router.post("/create_retriever")
async def create_retriever(
        request: LlamaCloudRetrieverRequest,
        service: LlamaCloudRetrieversService = Depends(LlamaCloudRetrieversService)
):
    """ Endpoint to create a retriever in LlamaCloud.
    Args:
            request: The request body containing the project id.
            service: LlamaCloudRetrieversService = Depends(LlamaCloudRetrieversService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.create_retriever(project_id=request.project_id, retriever_name=request.retriever_name)

@router.get("/get_retriever/{retriever_id}")
async def get_retriever(
        retriever_id: str,
        service: LlamaCloudRetrieversService = Depends(LlamaCloudRetrieversService)
):
    """ Endpoint to get a retriever in LlamaCloud.
    Args:
            retriever_id: The id of the retriever.
            service: LlamaCloudRetrieversService = Depends(LlamaCloudRetrieversService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.get_retriever(retriever_id)

@router.get("/list_retrievers")
async def list_retrievers(
        service: LlamaCloudRetrieversService = Depends(LlamaCloudRetrieversService)
):
    """ Endpoint to list all retrievers in LlamaCloud.
    Args:
            service: LlamaCloudRetrieversService = Depends(LlamaCloudRetrieversService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.list_retrievers()

@router.post("/create_chat_app")
async def create_chat_app(
        request: LlamaCloudChatAppRequest,
        service: LlamaCloudChatAppService = Depends(LlamaCloudChatAppService)
):
    """ Endpoint to create a chat app in LlamaCloud.
    Args:
            request: The request body containing the retriever id and chat app name.
            service: LlamaCloudChatAppService = Depends(LlamaCloudChatAppService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.create_chat_app_func(
        name=request.name,
        retriever_id=request.retriever_id
    )

@router.get("/get_chat_apps")
async def get_chat_apps(
        service: LlamaCloudChatAppService = Depends(LlamaCloudChatAppService)
):
    """ Endpoint to get all chat apps in LlamaCloud.
    Args:
            service: LlamaCloudChatAppService = Depends(LlamaCloudChatAppService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.get_chat_apps()

@router.delete("/delete_file/{file_id}")
async def delete_file(
        file_id: str,
        service: LlamaCloudFileService = Depends(LlamaCloudFileService)
):
    """ Endpoint to delete a file in LlamaCloud.
    Args:
            file_id: The id of the file.
            service: LlamaCloudFileService = Depends(LlamaCloudFileService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.delete_file(file_id)

@router.get("/llama_cloud/list_data_sources")
async def list_data_sources(
        service: LlamaCloudDataSinkService = Depends(LlamaCloudDataSinkService)
):
    """ Endpoint to list all data sources in LlamaCloud.
    Args:
            service: LlamaCloudDataSinkService = Depends(LlamaCloudDataSinkService)
    Returns:
        The response from LlamaCloud.
    """
    return await service.list_data_sinks()
