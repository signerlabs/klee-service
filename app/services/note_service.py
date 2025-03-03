import logging
import os
import time
import uuid
from typing import List

from sqlalchemy import select, or_, false, true

from app.model.Response import ResponseContent
from app.model.note import CreateNoteRequest, Note, NoteResponse
from app.services.client_sqlite_service import db_transaction
from app.model.klee_settings import Settings as KleeSettings
from app.services.llama_cloud.llama_cloud_file_service import LlamaCloudFileService
from app.services.llama_index_service import LlamaIndexService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoteService:
    def __init__(self):
        self.save_dir = f"{KleeSettings.temp_file_url}"
        self.llama_index_service = LlamaIndexService()
        self.llama_cloud_file_service = LlamaCloudFileService()

    @db_transaction
    async def create_note(
            self,
            request: CreateNoteRequest,
            session = None
    ) -> NoteResponse:
        """
        Create a note
        :param request: CreateNoteRequest
        :param session: session
        :return: NoteResponse
        """
        try:
            current_time = time.time()
            note_id = str(uuid.uuid4())

            local_mode = True
            if KleeSettings.local_mode is True:
                # Setting a temp file to store the note content
                save_file_path = f"{KleeSettings.temp_file_url}{note_id}" + "/store.txt"
                if not os.path.exists(f"{KleeSettings.temp_file_url}{note_id}"):
                    os.makedirs(f"{KleeSettings.temp_file_url}{note_id}", exist_ok=True)
                with open(save_file_path, "w", encoding="utf-8") as file:
                    file.write(request.content)

                vector_store_path = f"{KleeSettings.vector_url}{note_id}"
                if not os.path.exists(vector_store_path):
                    os.makedirs(vector_store_path, exist_ok=True)
                await self.llama_index_service.persist_file_to_disk_2(f"{KleeSettings.temp_file_url}{note_id}",
                                                                      vector_store_path)
            else:
                # Upload the note content to LlamaCloud
                save_file_path = f"{KleeSettings.temp_file_url}{note_id}" + f"/{note_id}.txt"
                if not os.path.exists(f"{KleeSettings.temp_file_url}{note_id}"):
                    os.makedirs(f"{KleeSettings.temp_file_url}{note_id}", exist_ok=True)
                with open(save_file_path, "w", encoding="utf-8") as file:
                    file.write(request.content)

                cloud_file = await self.llama_cloud_file_service.upload_file(f"{KleeSettings.temp_file_url}{note_id}/{note_id}.txt")
                note_id = cloud_file.id
                local_mode = False

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
            logger.error(f"note_embedding: {e}")
            raise e

    @db_transaction
    async def get_all_notes(
            self,
            keyword: str = None,
            session = None
    ) -> List[NoteResponse]:
        """
        Get all notes
        :param session: session
        :param keyword: str
        :return: List[NoteResponse]
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
            if KleeSettings.local_mode is False:
                query = query.filter(Note.local_mode == false())
            else:
                query = query.filter(Note.local_mode == true())

            result = await session.execute(query)
            notes = result.scalars().all()
            return [NoteResponse(**note.__dict__) for note in notes]
        except Exception as e:
            logger.error(f"get_all_notes: {e}")
            raise e

    @db_transaction
    async def update_note(
            self,
            note_id: str,
            request: CreateNoteRequest,
            session
    ) -> NoteResponse:
        """
        Update a note
        :param session: session
        :param note_id: str
        :param request: CreateNoteRequest
        :return: NoteResponse
        """
        try:
            result = await session.execute(select(Note).filter(Note.id == note_id))
            note = result.scalar_one_or_none()
            if note is None:
                logger.error(f"update_note: note_id: {note_id} not found")
                raise Exception(f"update_note: note_id: {note_id} not found")

            note.folder_id = request.folder_id
            note.title = request.title
            note.content = request.content
            note.type = request.type
            note.status = request.status
            note.is_pin = request.is_pin
            note.update_at = time.time()
            note.html_content = request.html_content

            if KleeSettings.local_mode is True:
                save_file_path = f"{KleeSettings.temp_file_url}{note_id}" + "/store.txt"
                with open(save_file_path, "w", encoding="utf-8") as file:
                    file.write(request.html_content)

                vector_store_path = f"{KleeSettings.vector_url}{note_id}"
                await self.llama_index_service.persist_file_to_disk_2(f"{KleeSettings.temp_file_url}{note_id}",
                                                                 vector_store_path)

            await session.flush()
            return NoteResponse(**note.__dict__)
        except Exception as e:
            logger.error(f"update_note: {e}")
            raise e

    @db_transaction
    async def delete_note(
            self,
            note_id: str,
            session
    ) -> ResponseContent:
        try:
            result = await session.execute(select(Note).filter(Note.id == note_id))
            note = result.scalar_one_or_none()

            await session.delete(note)

            if KleeSettings.local_mode is False:
                await self.llama_cloud_file_service.delete_file(note_id)

            return ResponseContent(error_code=0, message="Delete note successfully", data=None)
        except Exception as e:
            logger.error(f"delete_note: {e}")
            raise e

    @db_transaction
    async def get_note_by_id(
            self,
            note_id: str,
            session
    ) -> NoteResponse:
        """
        Get a note by id
        :param note_id: str
        :param session: session
        :return: NoteResponse
        """
        try:
            result = await session.execute(select(Note).filter(Note.id == note_id))
            note = result.scalar_one_or_none()
            if note is None:
                logger.error(f"get_note_by_id: note_id: {note_id} not found")
                raise Exception(f"get_note_by_id: note_id: {note_id} not found")
            return NoteResponse(**note.__dict__)
        except Exception as e:
            logger.error(f"get_note_by_id: {e}")
            raise e
