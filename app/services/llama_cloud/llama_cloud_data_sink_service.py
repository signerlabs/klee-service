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

    async def create_data_sink(self):
        response = await self.async_client.data_sinks.create_data_sink()
        return response

    async def list_data_sinks(self):
        response = await self.async_client.data_sinks.list_data_sinks()
        return response
