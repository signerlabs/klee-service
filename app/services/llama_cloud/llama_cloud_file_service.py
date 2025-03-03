import logging

from pydantic import BaseModel

from app.model.klee_settings import Settings as KleeSettings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaCloudFileRequest(BaseModel):
    file_path: str = None
    project_id: str = None
    organization_id: str = None


class LlamaCloudFileService:
    def __init__(self):
        self.async_client = KleeSettings.async_llama_cloud

    async def upload_file(
            self,
            file_path: str,
            project_id: str = None,
            external_file_id: str = None
    ):
        """
        Asynchronous function to upload a file to LlamaCloud.
        Args:
            project_id: The ID of the project to upload the file to.
            file_path: The path of the file to upload.
            external_file_id: The external ID of the file.
        Returns:
            None
        """
        with open(file_path, 'rb') as f:
            file = await self.async_client.files.upload_file(project_id=project_id, upload_file=f, external_file_id=external_file_id)
            logger.info(f"File uploaded: {file}")
            return file

    async def get_files(
            self
    ):
        """
        Asynchronous function to list all files in LlamaCloud.
        Returns:
            A list of all files in LlamaCloud.
        """
        return await self.async_client.files.list_files()

    async def delete_file(
            self,
            file_id: str
    ):
        """
        Asynchronous function to delete a file in LlamaCloud.
        Args:
            file_id: The ID of the file to delete.
        Returns:
            None
        """
        return await self.async_client.files.delete_file(id=file_id)

    async def read_file_content(
            self,
            file_id: str
    ):
        """
        Asynchronous function to read the content of a file in LlamaCloud.
        Args:
            file_id: The ID of the file to read.
        Returns:
            The content of the file in LlamaCloud.
        """
        content = await self.async_client.files.read_file_content(id=file_id)
        return content

    async def sync_file(
            self,
            request: LlamaCloudFileRequest
    ):
        """
        Asynchronous function to synchronize files in LlamaCloud.
        Args:
            request: The request object containing the project and organization IDs.
        Returns:
            None
        """
        await self.async_client.files.sync_files(
            project_id=request.project_id,
            organization_id=request.organization_id
        )

    async def get_file(
            self,
            file_id: str
    ):
        """
        Asynchronous function to get a file in LlamaCloud.
        Args:
            file_id: The ID of the file to get.
        Returns:
            The file in LlamaCloud.
        """
        return await self.async_client.files.get_file(id=file_id)
