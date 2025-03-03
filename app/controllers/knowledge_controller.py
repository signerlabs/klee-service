import logging
import shutil
from typing import Optional

from fastapi import APIRouter, Depends

from app.model.Response import ResponseContent
from app.model.knowledge import KnowledgeCreate, KnowledgeResponse
from app.services.llama_index_service import LlamaIndexService
from app.model.LlamaRequest import LlamaFileList, LLamaFileImportRequest

from app.services.knowledge_service import KnowledgeService

router = APIRouter()

logger = logging.getLogger(__name__)

llama_index_service = LlamaIndexService()

class KnowledgeController:
    def __init__(self):
        logger.info("init KnowledgeController")
        self.knowledge_service = KnowledgeService()

    async def create_knowledge(
            self,
            knowledge: KnowledgeCreate
    ):
        return await self.knowledge_service.create_knowledge(knowledge)

    async def llama_add(
            self,
            knowledge_id: str,
            file_obj: LlamaFileList
    ):
        return await self.knowledge_service.llama_add(knowledge_id, file_obj)

    async def update_knowledge(
            self,
            knowledge_id: str,
            knowledge: KnowledgeCreate
    ) -> KnowledgeResponse:
        return await self.knowledge_service.update_knowledge(knowledge_id, knowledge)

    async def delete_knowledge(
            self,
            knowledge_id: str,
    ):
        return await self.knowledge_service.delete_knowledge(knowledge_id)

    async def refresh_knowledge(
            self,
            knowledge_id: str,
            path: str
    ):
        return await self.knowledge_service.refresh_knowledge(knowledge_id, path)

    async def import_knowledge(
            self,
            knowledge_id: str,
            file_import: LLamaFileImportRequest
    ):
        return await self.knowledge_service.import_knowledge(knowledge_id, file_import)

    async def delete_file(
            self,
            file_id: str
    ):
        return await self.knowledge_service.delete_file(file_id)

@router.get('/')
async def get_all_knowledge(
    keyword: Optional[str] = None,
    service: KnowledgeService = Depends(KnowledgeService)
) -> ResponseContent:
    """
    Get all knowledge database.
    :param keyword: str, optional, keyword for search.
    :param service: KnowledgeService, service for knowledge.
    :return: ResponseContent, response content.
    """
    return await service.get_all_knowledge(keyword)


@router.get('/{knowledge_id}')
async def get_knowledge(
        knowledge_id: str,
        service: KnowledgeService = Depends(KnowledgeService),
):
    return await service.get_knowledge(knowledge_id)

@router.get("/all/{knowledge_id}")
async def get_all_files(
        knowledge_id: str,
        service: KnowledgeService = Depends(KnowledgeService),
):
    return await service.get_all_files(knowledge_id)

# 创建Knowledge知识库
@router.post('/')
async def create_knowledge(
        knowledge: KnowledgeCreate,
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.create_knowledge(knowledge)


@router.post("/llama/add/files/{knowledge_id}")
async def llama_add(
        knowledge_id: str,
        file_obj: LlamaFileList,
        controller: KnowledgeController = Depends(KnowledgeController)
):
    return await controller.llama_add(knowledge_id, file_obj)


async def write_file(
        src: str,
        dest: str
):
    shutil.copy(src, dest)


# 更新Knowledge知识库
@router.put('/{knowledge_id}', response_model=KnowledgeResponse)
async def update_knowledge(
        knowledge_id: str,
        knowledge: KnowledgeCreate,
        controller: KnowledgeController = Depends(KnowledgeController),
) -> KnowledgeResponse:
    return await controller.update_knowledge(knowledge_id, knowledge)


# 删除Knowledge知识库
@router.delete('/{knowledge_id}')
async def delete_knowledge(
        knowledge_id: str,
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.delete_knowledge(knowledge_id)


@router.get("/refresh/{knowledge_id}")
async def refresh_knowledge(
        knowledge_id: str,
        path: str,
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.refresh_knowledge(knowledge_id, path)


@router.post("/import/{knowledge_id}")
async def import_knowledge(
        knowledge_id: str,
        file_import: LLamaFileImportRequest,
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.import_knowledge(knowledge_id, file_import)


@router.delete("/file/{file_id}")
async def delete_file(
        file_id: str,
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.delete_file(file_id)
