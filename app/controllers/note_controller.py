import logging
from typing import Optional, List, Any, Coroutine

from fastapi import APIRouter, Depends, HTTPException, status, Header

from app.model.note import CreateNoteRequest, NoteResponse
from app.model.Response import ResponseContent
from app.services.note_service import NoteService, NoteServiceException, NoteNotFoundException, UnauthorizedException

router = APIRouter()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoteController:
    def __init__(self, note_service: NoteService = Depends(NoteService)):
        logger.info("NoteController initialized")
        self.note_service = note_service

    async def handle_note_exception(self, e: Exception) -> ResponseContent:
        """Unified note exception handler"""
        if isinstance(e, NoteNotFoundException):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        elif isinstance(e, UnauthorizedException):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
        elif isinstance(e, NoteServiceException):
            return ResponseContent(error_code=-1, message=str(e), data=None)
        else:
            logger.error(f"Unexpected error: {str(e)}")
            return ResponseContent(error_code=-1, message="Internal Server Error", data=None)

    async def get_all_notes(
            self,
            token,
            keyword: Optional[str] = None
    ):
        """Get all notes
        
        Args:
            keyword: Search keyword
            token: JWT token
        Returns:
            List of notes
        """
        try:
            return await self.note_service.get_all_notes(token=token, keyword=keyword)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def get_note_by_id(
            self,
            token,
            note_id: str
    ):
        """Get note by ID
        
        Args:
            note_id: Note ID
            token: JWT token
        Returns:
            Note details
        """
        try:
            return await self.note_service.get_note_by_id(token=token, note_id=note_id)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def create_note(
            self,
            token,
            request: CreateNoteRequest,
    ):
        """Create new note
        
        Args:
            token: JWT token
            request: Create note request
            
        Returns:
            Created note
        """
        try:
            return await self.note_service.create_note(
                token=token,
                request=request
            )
        except Exception as e:
            return await self.handle_note_exception(e)

    async def update_note(
            self,
            token,
            note_id: str,
            request: CreateNoteRequest,
    ):
        """Update note
        
        Args:
            note_id: Note ID
            request: Update note request
            token: JWT token
        Returns:
            Updated note
        """
        try:
            return await self.note_service.update_note(token=token, note_id=note_id, request=request)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def synchronize_files(
            self,
            token,
    ) -> ResponseContent:
        """Synchronize note files

        Args:
            token: JWT token
        Returns:
            Synchronization result
        """
        try:
            return await self.note_service.sync_note_files(token=token)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def delete_note(
            self,
            token,
            note_id: str
    ) -> ResponseContent:
        """Delete note
        
        Args:
            note_id: Note ID
            token: JWT token
        Returns:
            Delete result
        """
        try:
            return await self.note_service.delete_note(token=token, note_id=note_id)
        except Exception as e:
            return await self.handle_note_exception(e)


# API Route Handlers
@router.get("/")
async def get_all_notes(
        keyword: Optional[str] = None,
        Authorization: str = Header(None),
        controller: NoteController = Depends()
):
    """Get all notes"""
    return await controller.get_all_notes(token=Authorization, keyword=keyword)

@router.get("/synchronize-files")
async def synchronize_files(
        Authorization: str = Header(None),
        controller: NoteController = Depends()
) -> ResponseContent:
    """Synchronize note files"""
    logger.info(f"Synchronizing note files")
    return await controller.synchronize_files(token=Authorization)

@router.get("/{note_id}")
async def get_note_by_id(
        note_id: str,
        Authorization: str = Header(None),
        controller: NoteController = Depends()
):
    """Get note by ID"""
    return await controller.get_note_by_id(
        token=Authorization,
        note_id=note_id
    )


@router.post("/", response_model=NoteResponse)
async def create_note(
        request: CreateNoteRequest,
        Authorization: str = Header(None),
        controller: NoteController = Depends()
):
    """Create new note"""
    return await controller.create_note(
        token=Authorization,
        request=request
    )

@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(
        note_id: str,
        request: CreateNoteRequest,
        Authorization: str = Header(None),
        controller: NoteController = Depends()
):
    """Update note"""
    return await controller.update_note(token=Authorization, note_id=note_id, request=request)

@router.delete("/{note_id}")
async def delete_note(
        note_id: str,
        Authorization: str = Header(None),
        controller: NoteController = Depends()
) -> ResponseContent:
    """Delete note"""
    return await controller.delete_note(token=Authorization, note_id=note_id)

