import logging
import os
import time
import uuid
from typing import List, Optional

from sqlalchemy import select, or_, false, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.model.Response import ResponseContent
from app.model.note import CreateNoteRequest, Note, NoteResponse
from app.services.client_sqlite_service import db_transaction
from app.model.klee_settings import Settings as KleeSettings
from app.services.llama_cloud.llama_cloud_file_service import LlamaCloudFileService
from app.services.llama_index_service import LlamaIndexService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoteServiceException(Exception):
    """Base exception for NoteService"""
    pass


class NoteNotFoundException(NoteServiceException):
    """Raised when a note is not found"""
    pass


class NoteService:
    def __init__(self):
        self.save_dir = KleeSettings.temp_file_url
        self.vector_dir = KleeSettings.vector_url
        self.llama_index_service = LlamaIndexService()
        self.llama_cloud_file_service = LlamaCloudFileService()

    async def _save_local_file(self, note_id: str, content: str, is_store: bool = True) -> str:
        """Save content to local file system

        Args:
            note_id: The ID of the note
            content: The content to save
            is_store: Whether this is a store file or regular file

        Returns:
            The path where the file was saved
        """
        note_dir = os.path.join(self.save_dir, note_id)
        os.makedirs(note_dir, exist_ok=True)
        
        filename = "store.txt" if is_store else f"{note_id}.txt"
        file_path = os.path.join(note_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(content)
        
        return file_path

    async def _process_local_note(self, note_id: str, content: str) -> None:
        """Process note in local mode"""
        file_path = await self._save_local_file(note_id, content)
        vector_store_path = os.path.join(self.vector_dir, note_id)
        os.makedirs(vector_store_path, exist_ok=True)
        await self.llama_index_service.persist_file_to_disk_2(
            os.path.dirname(file_path),
            vector_store_path
        )

    async def _process_cloud_note(self, note_id: str, content: str) -> str:
        """Process note in cloud mode"""
        file_path = await self._save_local_file(note_id, content, False)
        cloud_file = await self.llama_cloud_file_service.upload_file(file_path)
        return cloud_file.id

    @db_transaction
    async def create_note(
            self,
            request: CreateNoteRequest,
            session: AsyncSession
    ) -> NoteResponse:
        """Create a new note

        Args:
            request: The note creation request
            session: The database session

        Returns:
            NoteResponse object containing the created note data

        Raises:
            NoteServiceException: If there's an error creating the note
        """
        try:
            current_time = time.time()
            note_id = str(uuid.uuid4())
            local_mode = KleeSettings.local_mode

            if local_mode:
                await self._process_local_note(note_id, request.content)
            else:
                note_id = await self._process_cloud_note(note_id, request.content)

            new_note = Note(
                id=note_id,
                folder_id=request.folder_id,
                title=request.title,
                content=request.content,
                type=request.type.note,
                status=request.status.normal,
                is_pin=request.is_pin,
                create_at=current_time,
                update_at=current_time,
                delete_at=0,
                html_content=request.html_content,
                local_mode=local_mode
            )

            session.add(new_note)
            await session.flush()

            return NoteResponse(**new_note.__dict__)
        except Exception as e:
            logger.error(f"Error creating note: {str(e)}")
            raise NoteServiceException(f"Failed to create note: {str(e)}") from e

    @db_transaction
    async def get_all_notes(
            self,
            keyword: Optional[str] = None,
            session: AsyncSession = None
    ) -> List[NoteResponse]:
        """Get all notes, optionally filtered by keyword

        Args:
            keyword: Optional search keyword to filter notes
            session: The database session

        Returns:
            List of NoteResponse objects

        Raises:
            NoteServiceException: If there's an error retrieving notes
        """
        try:
            query = select(Note)
            if keyword:
                query = query.filter(
                    or_(
                        Note.content.ilike(f"%{keyword}%"),
                        Note.title.ilike(f"%{keyword}%")
                    )
                )
            
            query = query.filter(
                Note.local_mode == (true() if KleeSettings.local_mode else false())
            )

            result = await session.execute(query)
            notes = result.scalars().all()
            return [NoteResponse(**note.__dict__) for note in notes]
        except Exception as e:
            logger.error(f"Error retrieving notes: {str(e)}")
            raise NoteServiceException(f"Failed to retrieve notes: {str(e)}") from e

    @db_transaction
    async def update_note(
            self,
            note_id: str,
            request: CreateNoteRequest,
            session: AsyncSession
    ) -> NoteResponse:
        """Update an existing note

        Args:
            note_id: The ID of the note to update
            request: The note update request
            session: The database session

        Returns:
            NoteResponse object containing the updated note data

        Raises:
            NoteNotFoundException: If the note is not found
            NoteServiceException: If there's an error updating the note
        """
        try:
            result = await session.execute(select(Note).filter(Note.id == note_id))
            note = result.scalar_one_or_none()
            if note is None:
                raise NoteNotFoundException(f"Note with ID {note_id} not found")

            note.folder_id = request.folder_id
            note.title = request.title
            note.content = request.content
            note.type = request.type
            note.status = request.status
            note.is_pin = request.is_pin
            note.update_at = time.time()
            note.html_content = request.html_content

            if KleeSettings.local_mode:
                await self._process_local_note(note_id, request.html_content)

            await session.flush()
            return NoteResponse(**note.__dict__)
        except NoteNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error updating note {note_id}: {str(e)}")
            raise NoteServiceException(f"Failed to update note: {str(e)}") from e

    @db_transaction
    async def delete_note(
            self,
            note_id: str,
            session: AsyncSession
    ) -> ResponseContent:
        """Delete a note

        Args:
            note_id: The ID of the note to delete
            session: The database session

        Returns:
            ResponseContent indicating success

        Raises:
            NoteNotFoundException: If the note is not found
            NoteServiceException: If there's an error deleting the note
        """
        try:
            result = await session.execute(select(Note).filter(Note.id == note_id))
            note = result.scalar_one_or_none()
            
            if note is None:
                raise NoteNotFoundException(f"Note with ID {note_id} not found")

            await session.delete(note)

            if not KleeSettings.local_mode:
                await self.llama_cloud_file_service.delete_file(note_id)

            return ResponseContent(
                error_code=0,
                message="Note deleted successfully",
                data=None
            )
        except NoteNotFoundException:
            raise
        except Exception as e:
            logger.error(f"Error deleting note {note_id}: {str(e)}")
            raise NoteServiceException(f"Failed to delete note: {str(e)}") from e

    @db_transaction
    async def get_note_by_id(
            self,
            note_id: str,
            session: AsyncSession
    ) -> NoteResponse:
        """Get a note by id

        Args:
            note_id: The ID of the note to retrieve
            session: The database session

        Returns:
            NoteResponse object containing the note data

        Raises:
            NoteNotFoundException: If the note is not found
        """
        try:
            result = await session.execute(select(Note).filter(Note.id == note_id))
            note = result.scalar_one_or_none()
            if note is None:
                logger.error(f"Note not found with ID: {note_id}")
                raise NoteNotFoundException(f"Note with ID {note_id} not found")
            return NoteResponse(**note.__dict__)
        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {str(e)}")
            raise
