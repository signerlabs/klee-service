import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import List, Optional

import httpx
from sqlalchemy import select, or_, false, true
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.env_config import config
from app.model.Response import ResponseContent
from app.model.knowledge import File
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


class UnauthorizedException(NoteServiceException):
    """Raised when authentication fails with 401 status code"""
    pass


class NoteService:
    def __init__(self):
        logger.info("NoteService initialized")
        self.save_dir = KleeSettings.temp_file_url
        self.vector_dir = KleeSettings.vector_url
        self.llama_index_service = LlamaIndexService()
        self.llama_cloud_file_service = LlamaCloudFileService()

    async def _handle_response(self, response, not_found_message=None):
        """
        Handle HTTP response and check for common error status codes
        
        Args:
            response: The HTTP response object
            not_found_message: Custom message for 404 errors
            
        Raises:
            UnauthorizedException: If status code is 401
            NoteNotFoundException: If status code is 404
        """
        if response.status_code == 401:
            logger.error(f"Unauthorized access: {response.url}")
            raise UnauthorizedException("Unauthorized access: Invalid or expired token")
        
        if response.status_code == 404 and not_found_message:
            raise NoteNotFoundException(not_found_message)
            
        response.raise_for_status()
        return response.json()

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
            token,
            request: CreateNoteRequest,
            session: AsyncSession
    ):
        """Create a new note

        Args:
            token: The user token
            request: The note creation request
            session: The database session

        Returns:
            NoteResponse object containing the created note data

        Raises:
            NoteServiceException: If there's an error creating the note
        """
        try:
            if KleeSettings.local_mode is True:
                current_time = time.time()
                note_id = str(uuid.uuid4())
                local_mode = KleeSettings.local_mode

                if local_mode:
                    await self._process_local_note(note_id, request.content)

                new_note = Note(
                    id=note_id,
                    folder_id=request.folder_id,
                    title=request.title,
                    content=request.content,
                    type="note",
                    status="normal",
                    is_pin=request.is_pin,
                    create_at=current_time,
                    update_at=current_time,
                    delete_at=0,
                    html_content=request.html_content,
                    local_mode=local_mode
                )

                session.add(new_note)
                await session.flush()

                note_dict = {k: v for k, v in new_note.__dict__.items() if k != '_sa_instance_state'}
                return note_dict
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{config.klee_cloud_api_url}/note/create",
                        headers={"Authorization": f"Bearer {token}"}, json=asdict(request)
                    )
                    response.raise_for_status()
                    logger.info(f"Create note response: {response.json()}")

                    await self._process_local_note(response.json()["id"], request.content)
                    return response.json()
        except Exception as e:
            logger.error(f"Error creating note: {str(e)}")
            raise NoteServiceException(f"Failed to create note: {str(e)}") from e

    @db_transaction
    async def get_all_notes(
            self,
            token: str,
            keyword: Optional[str] = None,
            session: AsyncSession = None
    ):
        """Get all notes, optionally filtered by keyword

        Args:
            keyword: Optional search keyword to filter notes
            session: The database session
            token: The user token
        Returns:
            List of NoteResponse objects

        Raises:
            NoteServiceException: If there's an error retrieving notes
        """
        try:
            if KleeSettings.local_mode is True:
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

                return_list = []

                for note in notes:
                    note_dict = {k: v for k, v in note.__dict__.items() if k != '_sa_instance_state'}
                    return_list.append(note_dict)

                return return_list
            else:
                response = await KleeSettings.async_http_client.get(
                    f"{config.klee_cloud_api_url}/note/all?keyword={keyword}",
                    headers={"Authorization": f"Bearer {token}"},
                )
                response.raise_for_status()
                logger.info(f"Get all notes response: {response.json()}")
                return response.json()
        except Exception as e:
            logger.error(f"Error retrieving notes: {str(e)}")
            raise NoteServiceException(f"Failed to retrieve notes: {str(e)}") from e

    @db_transaction
    async def update_note(
            self,
            token,
            note_id: str,
            request: CreateNoteRequest,
            session: AsyncSession
    ):
        """Update an existing note

        Args:
            token: The user token
            note_id: The ID of the note to update
            request: The update request containing new note data
            session: The database session

        Returns:
            Updated note data

        Raises:
            NoteNotFoundException: If the note is not found
            UnauthorizedException: If authentication fails (401)
            NoteServiceException: If there's an error updating the note
        """
        try:
            if KleeSettings.local_mode is True:
                result = await session.execute(select(Note).filter(Note.id == note_id))
                note = result.scalar_one_or_none()

                if note is None:
                    raise NoteNotFoundException(f"Note with ID {note_id} not found")

                note.title = request.title
                note.content = request.content
                note.update_at = datetime.now().timestamp()

                await session.commit()
                await self._process_local_note(note_id, request.content)
                note_dict = {k: v for k, v in note.__dict__.items() if k != '_sa_instance_state'}
                return note_dict
            else:
                response = await KleeSettings.async_http_client.put(
                    f"{config.klee_cloud_api_url}/note/{note_id}",
                    headers={"Authorization": f"Bearer {token}"}, json=asdict(request)
                )
                result = await self._handle_response(
                    response, 
                    not_found_message=f"Note with ID {note_id} not found"
                )
                await self._process_local_note(result["id"], request.content)
                return result
        except (NoteNotFoundException, UnauthorizedException):
            raise
        except Exception as e:
            logger.error(f"Error updating note {note_id}: {str(e)}")
            raise NoteServiceException(f"Failed to update note: {str(e)}") from e

    @db_transaction
    async def delete_note(
            self,
            token,
            note_id: str,
            session: AsyncSession
    ) -> ResponseContent:
        """Delete a note

        Args:
            note_id: The ID of the note to delete
            session: The database session
            token: The user token
        Returns:
            ResponseContent indicating success

        Raises:
            NoteNotFoundException: If the note is not found
            NoteServiceException: If there's an error deleting the note
        """
        try:
            if KleeSettings.local_mode is True:
                result = await session.execute(select(Note).filter(Note.id == note_id))
                note = result.scalar_one_or_none()

                if note is None:
                    raise NoteNotFoundException(f"Note with ID {note_id} not found")

                await session.delete(note)

                return ResponseContent(
                    error_code=0,
                    message="Note deleted successfully",
                    data=None
                )
            else:
                response = await KleeSettings.async_http_client.delete(
                    f"{config.klee_cloud_api_url}/note/{note_id}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if response.status_code == 404:
                    raise NoteNotFoundException(f"Note with ID {note_id} not found")
                response.raise_for_status()
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
            token,
            note_id: str,
            session: AsyncSession
    ):
        """Get a note by id

        Args:
            token: The user token
            note_id: The ID of the note to retrieve
            session: The database session

        Returns:
            NoteResponse object containing the note data

        Raises:
            NoteNotFoundException: If the note is not found
            UnauthorizedException: If authentication fails (401)
        """
        try:
            if KleeSettings.local_mode is True:
                result = await session.execute(select(Note).filter(Note.id == note_id))
                note = result.scalar_one_or_none()
                if note is None:
                    logger.error(f"Note not found with ID: {note_id}")
                    raise NoteNotFoundException(f"Note with ID {note_id} not found")
                note_dict = {k: v for k, v in note.__dict__.items() if k != '_sa_instance_state'}
                return note_dict
            else:
                response = await KleeSettings.async_http_client.get(
                    f"{config.klee_cloud_api_url}/note/{note_id}",
                    headers={"Authorization": f"Bearer {token}"}
                )
                return await self._handle_response(
                    response, 
                    not_found_message=f"Note with ID {note_id} not found"
                )
        except (UnauthorizedException, NoteNotFoundException):
            raise
        except Exception as e:
            logger.error(f"Error retrieving note {note_id}: {str(e)}")
            raise

    async def sync_files(
            self,
            token,
            note_id: str,
    ):
        """
        Synchronize files for a note
        Args:
            note_id: The ID of the note to synchronize files for

        Returns:

        """
        try:
            if KleeSettings.local_mode is False:
                upload_file_response = await KleeSettings.async_http_client.post(
                    headers={"Authorization": f"Bearer {token}"},
                    url=f"{config.klee_cloud_api_url}/file/upload-temp",
                    files={"file": open(f"{KleeSettings.temp_file_url}{note_id}/store.txt", "rb")}
                )
                upload_file_response.raise_for_status()

                temp_path = upload_file_response.json()["temp_file_path"]
                filename = upload_file_response.json()["filename"]

                await KleeSettings.async_http_client.get(
                    f"{config.klee_cloud_api_url}/note/generate-presigned-url/{note_id}",
                    headers={"Authorization": f"Bearer {token}"}
                )
        except Exception as e:
            logger.error(f"Error synchronizing files for note {note_id}: {str(e)}")
            raise NoteServiceException(f"Failed to synchronize files: {str(e)}") from e

    async def sync_note_files(
            self,
            token
    ):
        """
        Synchronize files for all notes
        Returns:

        """
        try:
            logger.info(f"Service: Starting note files synchronization. local_mode={KleeSettings.local_mode}")
            if KleeSettings.local_mode is False:
                logger.info("Service: Running in cloud mode, proceeding with synchronization")
                notes = await self.get_all_notes(token=token)
                logger.info(f"Service: Found {len(notes)} notes to synchronize")
                for note in notes:
                    logger.info(f"Service: Synchronizing note {note['id']}")
                    new_file = File(
                        name=str(note["id"]) + ".txt",
                        os_mtime=datetime.now().timestamp(),
                        format="txt",
                        size=os.path.getsize(f"{KleeSettings.temp_file_url}{note['id']}/store.txt"),
                    )
                    # upload file to cloud
                    logger.info(f"Service: Uploading file for note {note['id']}")
                    upload_file_response = await KleeSettings.async_http_client.post(
                        headers={"Authorization": f"Bearer {token}"},
                        url=f"{config.klee_cloud_api_url}/file/upload-temp",
                        files={"file": open(f"{KleeSettings.temp_file_url}{note['id']}/store.txt", "rb")}
                    )
                    upload_file_response.raise_for_status()

                    new_file.path = upload_file_response.json()["temp_file_path"]
                    logger.info(f"Service: File uploaded successfully for note {note['id']}")

                    headers = {"Authorization": f"Bearer {token}"}
                    request_data = {
                        "note_id": note["id"],
                        "file_temp_path": new_file.path
                    }
                    logger.info(f"Service: Syncing note {note['id']} with cloud")
                    response = await KleeSettings.async_http_client.put(
                        headers=headers,
                        url=f"{config.klee_cloud_api_url}/note/sync-update/{request_data['note_id']}",
                        json=request_data
                    )
                    response.raise_for_status()
                    logger.info(f"Service: Note {note['id']} synchronized successfully")
                logger.info("Service: All notes synchronized successfully")
                return ResponseContent(
                    error_code=0,
                    message="Notes synchronized successfully",
                    data=None
                )
            else:
                logger.info("Service: Running in local mode, skipping synchronization")
                return ResponseContent(
                    error_code=0,
                    message="Running in local mode, synchronization skipped",
                    data=None
                )
        except Exception as e:
            logger.error(f"Service: Error synchronizing files for all notes: {str(e)}")
            raise NoteServiceException(f"Failed to synchronize files for all notes: {str(e)}") from e
