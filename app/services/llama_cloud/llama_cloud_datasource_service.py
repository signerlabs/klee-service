import logging
import os

from dotenv import load_dotenv
from llama_cloud.client import AsyncLlamaCloud

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaCloudDataSinkService:
    def __init__(self):
        load_dotenv(f".env")
        self.async_client = AsyncLlamaCloud(token=os.getenv("LLAMA_CLOUD_API_KEY"))

    async def list_data_sources(self):
        response = await self.async_client.data_sources.list_data_sources()
        return response