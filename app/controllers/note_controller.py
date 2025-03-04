import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.model.note import CreateNoteRequest, NoteResponse
from app.model.Response import ResponseContent
from app.services.note_service import NoteService, NoteServiceException, NoteNotFoundException

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
        elif isinstance(e, NoteServiceException):
            return ResponseContent(error_code=-1, message=str(e), data=None)
        else:
            logger.error(f"Unexpected error: {str(e)}")
            return ResponseContent(error_code=-1, message="Internal Server Error", data=None)

    async def get_all_notes(
            self,
            keyword: Optional[str] = None
    ) -> List[NoteResponse]:
        """Get all notes
        
        Args:
            keyword: Search keyword
            
        Returns:
            List of notes
        """
        try:
            return await self.note_service.get_all_notes(keyword)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def get_note_by_id(
            self,
            note_id: str
    ) -> NoteResponse:
        """Get note by ID
        
        Args:
            note_id: Note ID
            
        Returns:
            Note details
        """
        try:
            return await self.note_service.get_note_by_id(note_id)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def create_note(
            self,
            request: CreateNoteRequest,
    ) -> NoteResponse:
        """Create new note
        
        Args:
            request: Create note request
            
        Returns:
            Created note
        """
        try:
            return await self.note_service.create_note(request)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def update_note(
            self,
            note_id: str,
            request: CreateNoteRequest,
    ) -> NoteResponse:
        """Update note
        
        Args:
            note_id: Note ID
            request: Update note request
            
        Returns:
            Updated note
        """
        try:
            return await self.note_service.update_note(note_id, request)
        except Exception as e:
            return await self.handle_note_exception(e)

    async def delete_note(
            self,
            note_id: str
    ) -> ResponseContent:
        """Delete note
        
        Args:
            note_id: Note ID
            
        Returns:
            Delete result
        """
        try:
            return await self.note_service.delete_note(note_id)
        except Exception as e:
            return await self.handle_note_exception(e)


# API Route Handlers
@router.get("/", response_model=List[NoteResponse])
async def get_all_notes(
        keyword: Optional[str] = None,
        controller: NoteController = Depends()
) -> List[NoteResponse]:
    """Get all notes"""
    return await controller.get_all_notes(keyword)


@router.get("/{note_id}", response_model=NoteResponse)
async def get_note_by_id(
        note_id: str,
        controller: NoteController = Depends()
) -> NoteResponse:
    """Get note by ID"""
    return await controller.get_note_by_id(note_id)


@router.post("/", response_model=NoteResponse)
async def create_note(
        request: CreateNoteRequest,
        controller: NoteController = Depends()
) -> NoteResponse:
    """Create new note"""
    return await controller.create_note(request)


@router.put("/{note_id}", response_model=NoteResponse)
async def update_note(
        note_id: str,
        request: CreateNoteRequest,
        controller: NoteController = Depends()
) -> NoteResponse:
    """Update note"""
    return await controller.update_note(note_id, request)


@router.delete("/{note_id}")
async def delete_note(
        note_id: str,
        controller: NoteController = Depends()
) -> ResponseContent:
    """Delete note"""
    return await controller.delete_note(note_id)

