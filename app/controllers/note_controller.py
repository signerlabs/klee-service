import logging

from fastapi import APIRouter, Depends

from app.model.note import CreateNoteRequest, NoteResponse
from app.model.Response import ResponseContent
from app.services.note_service import NoteService

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoteController:
    def __init__(self):
        logger.info("NoteController initialized")
        self.note_service = NoteService()

    async def get_all_notes(
            self,
            keyword: str = None
    ):
        """
        Get All Notes
        :param keyword: keyword for search
        :return: list of notes
        :exception HTTPException: if error occurs
        """
        try:
            return await self.note_service.get_all_notes(keyword)
        except Exception as e:
            logger.error(f"Get All Notes Error: {str(e)}")
            return ResponseContent(error_code=-1, message="Get All Notes Error", data=None)

    async def get_note_by_id(
            self,
            note_id: str
    ):
        """
        Get note by id
        :param note_id: note id
        :return: NoteResponse
        :exception HTTPException: if error occurs
        """
        try:
            return await self.note_service.get_note_by_id(note_id)
        except Exception as e:
            logger.error(f"Get Note By Id Error: {str(e)}")
            return ResponseContent(error_code=-1, message="Get Note By Id Error", data=None)

    async def create_note(
            self,
            request: CreateNoteRequest,
    ):
        """
        Create Note
        :param request: CreateNoteRequest
        :return: NoteResponse
        :exception HTTPException: if error occurs
        """
        try:
            return await self.note_service.create_note(request)
        except Exception as e:
            logger.error(f"Create Note Error: {str(e)}")
            return ResponseContent(error_code=-1, message="Note Creation Failed", data=None)

    async def update_note(
            self,
            note_id: str,
            request: CreateNoteRequest,
    ):
        """
        Update Note
        :param note_id: note id
        :param request: CreateNoteRequest
        :return: NoteResponse
        :exception HTTPException: if error occurs
        """
        try:
            return await self.note_service.update_note(note_id, request)
        except Exception as e:
            logger.error(f"Update Note Error: {str(e)}")
            return ResponseContent(error_code=-1, message="笔记更新失败", data=None)

    async def delete_note(
            self,
            note_id: str
    ):
        """
        Delete Note
        :param note_id: note id
        :return: ResponseContent
        :exception HTTPException: if error occurs
        """
        try:
            return await self.note_service.delete_note(note_id)
        except Exception as e:
            logger.error(f"Delete Note Error: {str(e)}")
            return ResponseContent(error_code=-1, message="笔记删除失败", data=None)


@router.get("/")
async def get_all_notes(
        keyword: str = None,
        controller: NoteController = Depends(lambda: NoteController())
):
    """
    Get All Notes
    :param keyword: keyword for search
    :param controller: NoteController
    :return: list of notes
    """
    return await controller.get_all_notes(keyword)

@router.get("/{note_id}")
async def get_note_by_id(
        note_id: str,
        controller: NoteController = Depends(lambda: NoteController())
):
    """
    Get note by id
    :param note_id: note id
    :param controller: NoteController
    :return: NoteResponse
    """
    return await controller.get_note_by_id(note_id)

@router.post("/")
async def create_note(
        request: CreateNoteRequest,
        controller: NoteController = Depends(NoteController)
):
    """
    Create Note
    :param request: CreateNoteRequest
    :param controller: NoteController
    :return: NoteResponse
    """
    return await controller.create_note(request)


@router.put("/{note_id}")
async def update_note(
        note_id: str,
        request: CreateNoteRequest,
        controller: NoteController = Depends(lambda: NoteController())
):
    """
    Update Note
    :param note_id: note id
    :param request: CreateNoteRequest
    :param controller: NoteController
    :return: NoteResponse
    """
    return await controller.update_note(note_id, request)

@router.delete("/{note_id}")
async def delete_note(
        note_id: str,
        controller: NoteController = Depends(lambda: NoteController())
):
    """
    Delete Note
    :param note_id: note id
    :param controller: NoteController
    :return: ResponseContent
    """
    return await controller.delete_note(note_id)

