import logging
import shutil
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.params import Header

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

@router.get('/')
async def get_all_knowledge(
    keyword: Optional[str] = None,
    Authorization: str = Header(None),
    controller: KnowledgeController = Depends(KnowledgeController),
) -> ResponseContent:
    """
    Get all knowledge database.
    :param keyword: str, optional, keyword for search.
    :param Authorization: str, optional, JWT token.
    :param controller: KnowledgeController, controller for knowledge.
    :return: ResponseContent, response content.
    """
    logger.info(f"get_all_knowledge: keyword: {keyword}, Authorization: {Authorization}")
    return await controller.knowledge_service.get_all_knowledge(
        token=Authorization,
        keyword=keyword
    )


@router.get('/{knowledge_id}')
async def get_knowledge(
        knowledge_id: str,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
) -> ResponseContent:
    return await controller.knowledge_service.get_knowledge(
        token=Authorization,
        knowledge_id=knowledge_id
    )

@router.get("/all/{knowledge_id}")
async def get_all_files(
        knowledge_id: str,
        Authorization: str = Header(None),
        service: KnowledgeService = Depends(KnowledgeService),
):
    return await service.get_all_files(token=Authorization, knowledge_id=knowledge_id)

# 创建Knowledge知识库
@router.post('/')
async def create_knowledge(
        knowledge: KnowledgeCreate,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
):
    logger.info(f"create_knowledge: {knowledge}, Authorization: {Authorization}")
    return await controller.knowledge_service.create_knowledge(token=Authorization, knowledge=knowledge)


@router.post("/llama/add/files/{knowledge_id}")
async def llama_add(
        knowledge_id,
        file_obj: LlamaFileList,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController)
):
    return await controller.knowledge_service.llama_add(token=Authorization, knowledge_id=knowledge_id, file_obj=file_obj)


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
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
) -> KnowledgeResponse:
    return await controller.knowledge_service.update_knowledge(token=Authorization, knowledge_id=knowledge_id, knowledge=knowledge)


# 删除Knowledge知识库
@router.delete('/{knowledge_id}')
async def delete_knowledge(
        knowledge_id: str,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.knowledge_service.delete_knowledge(token=Authorization, knowledge_id=knowledge_id)


@router.get("/refresh/{knowledge_id}")
async def refresh_knowledge(
        knowledge_id: str,
        path: str,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.knowledge_service.refresh_knowledge(token=Authorization, knowledge_id=knowledge_id, path=path)


@router.post("/import/{knowledge_id}")
async def import_knowledge(
        knowledge_id: str,
        file_import: LLamaFileImportRequest,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.knowledge_service.import_knowledge(token=Authorization, knowledge_id=knowledge_id, file_import=file_import)


@router.delete("/file/{file_id}")
async def delete_file(
        file_id: str,
        Authorization: str = Header(None),
        controller: KnowledgeController = Depends(KnowledgeController),
):
    return await controller.knowledge_service.delete_file(token=Authorization, file_id=file_id)
