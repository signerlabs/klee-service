import logging
from typing import List

import httpx
import os

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatCloudService:
    def __init__(self):
        load_dotenv(f".env.dev")
        # Llama Cloud API Key
        self.llama_cloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        # Llama Cloud API URL for Knowledge Bases
        self.api_url = "https://api.llama-cloud.com/v1/knowledge_bases"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llama_cloud_api_key}",
        }
        self.timeout = httpx.Timeout(60.0)

    async def achat(
            self,
            message: str,
            knowledge_base_ids: List[str]
    ):
        logger.info(f"Sending message to Llama Cloud: {message}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url=self.api_url,
                headers=self.headers,
                json= {
                    "query": message,
                    "knowledge_bases": knowledge_base_ids,
                    "max_tokens": 1000,
                    "max_retries": 5,
                    "temperature": 0.5,
                    "top_k": 5
                }
            )
            response.raise_for_status()
            logger.info(f"Async response from Llama Cloud: {response.json()}")
            return response.json()
