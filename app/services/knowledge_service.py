# coding:utf8
import os
import logging
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import select, or_, update, delete, true, false
from starlette import status
from starlette.exceptions import HTTPException

from app.common.LlamaEnum import SystemTypeDiff
from app.model.Response import ResponseContent
from app.model.knowledge import File, Knowledge, KnowledgeCreate, EmbedStatus, KnowledgeResponse
from app.services.client_sqlite_service import db_transaction
from app.model.LlamaRequest import LlamaKnowledge, LlamaFileList, LLamaFileImportRequest
from app.services.llama_cloud.llama_cloud_file_service import LlamaCloudFileService

from app.services.llama_index_service import LlamaIndexService
from app.model.klee_settings import Settings as KleeSettings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self):
        self.llama_index_service = LlamaIndexService()
        self.llama_cloud_file_service = LlamaCloudFileService()

    @db_transaction
    async def get_all_knowledge(
            self,
            keyword: Optional[str] = None,
            session=None
    ) -> ResponseContent:
        """
        Get all knowledge database entries
        Args:
            keyword: Optional search keyword
            session: Database session
        Returns:
            ResponseContent: Response containing knowledge list
        """
        try:
            stmt = select(Knowledge).where(Knowledge.parent_id == '')
            if keyword:
                stmt = stmt.filter(
                    or_(
                        Knowledge.title.ilike(f"%{keyword}%"),
                        Knowledge.description.ilike(f"%{keyword}%")
                    )
                )
            if KleeSettings.local_mode is True:
                stmt = stmt.filter(Knowledge.local_mode == true())
            else:
                stmt = stmt.filter(Knowledge.local_mode == false())

            result = await session.execute(stmt)
            knowledge_list = result.scalars().all()

            knowledge_data = [
                asdict(k)
                for k in knowledge_list
            ]

            return ResponseContent(error_code=0, message="Successfully retrieved all knowledge entries", data=knowledge_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge entries: {str(e)}")

    @db_transaction
    async def get_knowledge(
            self,
            knowledge_id: str,
            session=None
    ):
        """
        Get a specific knowledge entry
        Args:
            knowledge_id: ID of the knowledge entry
            session: Database session
        Returns:
            ResponseContent: Response containing knowledge data
        """
        try:
            stmt = select(Knowledge).where(Knowledge.id == knowledge_id)
            result = await session.execute(stmt)
            knowledge = result.scalars().first()

            if not knowledge:
                raise HTTPException(status_code=404, detail="Knowledge entry not found")

            knowledge_dict = asdict(knowledge)

            return ResponseContent(error_code=0, message="Successfully retrieved knowledge entry", data=knowledge_dict)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to retrieve knowledge entry: {str(e)}")

    @db_transaction
    async def get_all_files(
            self,
            knowledge_id: str,
            session=None
    ):
        """
        Get all files associated with a knowledge entry
        Args:
            knowledge_id: ID of the knowledge entry
            session: Database session
        Returns:
            ResponseContent: Response containing file list
        """
        try:
            stmt = select(File).where(File.knowledgeId == knowledge_id)
            result = await session.execute(stmt)
            files = result.scalars().all()

            if KleeSettings.local_mode is True:
                return_files = [self.file_to_dict(k) for k in files]
                return ResponseContent(error_code=0, message="Successfully retrieved files", data=return_files)
            else:
                return_files = []
                for file in files:
                    cloud_file = await self.llama_cloud_file_service.get_file(file_id=file.id)
                    logger.info(f"cloud_file: {cloud_file}")

                    return_file = {
                        "id": cloud_file.id,
                        "os_time": cloud_file.created_at,
                        "name": cloud_file.name,
                        "path": "",
                        "size": file.size,
                        "create_at": cloud_file.created_at,
                        "update_at": cloud_file.last_modified_at,
                        "knowledgeId": file.knowledgeId
                    }
                    return_files.append(return_file)
                return ResponseContent(error_code=0, message="Successfully retrieved files", data=return_files)

        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve files: {str(e)}")

    @db_transaction
    async def create_knowledge(
            self,
            knowledge: KnowledgeCreate,
            session=None
    ):
        """
        Create a new knowledge entry
        Args:
            knowledge: Knowledge creation data
            session: Database session
        Returns:
            ResponseContent: Response containing created knowledge data
        """
        try:
            new_knowledge = Knowledge(
                id=str(uuid.uuid4()),
                timeStamp=datetime.now().timestamp(),
                title=knowledge.title,
                icon=knowledge.icon,
                description=knowledge.description,
                category=knowledge.category,
                isPin=knowledge.isPin,
                folder_path=knowledge.folder_path,
                embed_status=EmbedStatus.EMBEDDING.value,
                create_at=datetime.now().timestamp(),
                update_at=datetime.now().timestamp(),
                local_mode=KleeSettings.local_mode
            )
            session.add(new_knowledge)

            if KleeSettings.local_mode is True:
                vector_url = f"{KleeSettings.vector_url}{new_knowledge.id}"
                if not os.path.exists(vector_url):
                    os.makedirs(vector_url, exist_ok=True)

                temp_file_url = f"{KleeSettings.temp_file_url}{new_knowledge.id}"
                if not os.path.exists(temp_file_url):
                    os.makedirs(temp_file_url, exist_ok=True)
                temp_file_url += "/store.txt"
                with open(temp_file_url, "w", encoding="utf-8") as file:
                    file.write("")

            knowledge_response = asdict(new_knowledge)
            return ResponseContent(error_code=0, message="Successfully created knowledge entry", data=knowledge_response)
        except Exception as e:
            logger.error(f"Create knowledge error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create knowledge entry: {str(e)}")

    @db_transaction
    async def update_knowledge(
            self,
            knowledge_id: str,
            knowledge: KnowledgeCreate,
            session=None
    ) -> KnowledgeResponse:
        try:
            stmt = select(Knowledge).where(Knowledge.id == knowledge_id).with_for_update()
            result = await session.execute(stmt)
            existing_knowledge = result.scalar_one_or_none()

            if not existing_knowledge:
                raise HTTPException(status_code=404, detail="Database knowledge not found")

            update_stmt = (
                update(Knowledge)
                .where(Knowledge.id == knowledge_id)
                .values(
                    title=knowledge.title,
                    icon=knowledge.icon,
                    description=knowledge.description,
                    category=knowledge.category,
                    isPin=knowledge.isPin,
                    folder_path=knowledge.folder_path
                )
                .returning(Knowledge)
            )
            result = await session.execute(update_stmt)
            updated_knowledge = result.scalar_one()
            return KnowledgeResponse.from_orm(updated_knowledge)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Update knowledge database failed: {str(e)}")

    @db_transaction
    async def delete_knowledge(
            self,
            knowledge_id: str,
            session=None
    ):
        """
        Delete a knowledge entry and its associated files
        Args:
            knowledge_id: ID of the knowledge entry to delete
            session: Database session
        Returns:
            ResponseContent: Response indicating success/failure
        """
        try:
            stmt = select(Knowledge).where(Knowledge.id == knowledge_id)
            result = await session.execute(stmt)
            knowledge = result.scalar_one_or_none()

            if not knowledge:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Knowledge entry not found")

            await session.delete(knowledge)

            if KleeSettings.local_mode is False:
                stmt_file = select(File).where(File.knowledgeId == knowledge_id)
                result_file = await session.execute(stmt_file)
                files = result_file.scalars().all()
                for file in files:
                    await self.llama_cloud_file_service.delete_file(file_id=file.id)

            return ResponseContent(error_code=0, message="Successfully deleted knowledge entry", data=None)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete knowledge entry: {str(e)}")

    @db_transaction
    async def refresh_knowledge(
            self,
            knowledge_id: str,
            path: str,
            session=None
    ):
        try:
            import_files = []
            dir_path = Path(path)
            for file in dir_path.rglob('*'):
                if file.is_file():
                    file_path = str(file)
                    if file_path.find("DS_Store") != -1:
                        continue
                    title = ""
                    if KleeSettings.os_type == SystemTypeDiff.WIN.value:
                        title = file_path.split("\\")[-1]
                    elif KleeSettings.os_type == SystemTypeDiff.MAC.value:
                        title = file_path.split("/")[-1]
                    file_size = os.path.getsize(file_path)
                    import_files.append({"path": file_path, "name": title, "size": file_size})

            stmt = select(File).where(File.knowledgeId == knowledge_id)
            results = await session.execute(stmt)
            files = results.scalars().all()

            new_files = []
            if len(import_files) > 0:
                for import_file in import_files:
                    flag = True
                    for file in files:
                        # 判断是否有这个文件
                        if import_file["path"] == file.path:
                            flag = False
                            # 判断文件是否发生变化
                            if import_file["size"] != file.size:
                                file.size = import_file['size']
                                # await persist_one_file_to_disk(import_file["path"], file.id)
                                if not os.path.exists(f"{KleeSettings.temp_file_url}{file.id}"):
                                    os.makedirs(f"{KleeSettings.temp_file_url}{file.id}", exist_ok=True)
                                await self.write_file(f"{import_file["path"]}",
                                                      f"{KleeSettings.temp_file_url}{file.id}/{file.name}")

                                await self.llama_index_service.persist_file_to_disk_2(
                                    path=f"{KleeSettings.temp_file_url}{file.id}",
                                    store_dir=f"{KleeSettings.vector_url}{file.id}"
                                )
                                flag = False
                                break
                    if flag is True:
                        file_id = str(uuid.uuid4())
                        now_time = datetime.now().timestamp()
                        new_file = File()
                        new_file.__dict__.update({
                            "id": file_id,
                            "name": import_file["name"],
                            "os_mtime": now_time,
                            "format": import_file["name"].split(".")[-1],
                            "size": import_file["size"],
                            "knowledgeId": knowledge_id,
                            "path": import_file["path"],
                            "create_at": now_time,
                            "update_at": now_time
                        })
                        if not os.path.exists(f"{KleeSettings.temp_file_url}{file_id}"):
                            os.makedirs(f"{KleeSettings.temp_file_url}{file_id}", exist_ok=True)
                        await self.write_file(f"{import_file["path"]}",
                                              f"{KleeSettings.temp_file_url}{file_id}/{new_file.name}")

                        await self.llama_index_service.persist_file_to_disk_2(
                            path=f"{KleeSettings.temp_file_url}{new_file.id}",
                            store_dir=f"{KleeSettings.vector_url}{new_file.id}"
                        )
                        new_files.append(new_file)

            delete_file_id = []
            if len(import_files) > 0:
                for file in files:
                    flag = True
                    for import_file in import_files:
                        if file.path == import_file["path"]:
                            flag = False
                            break
                    if flag is True:
                        delete_file_id.append(file.id)

            session.add_all(files)
            session.add_all(new_files)

            delete_stmt = delete(File).where(File.id.in_(delete_file_id))

            for i in delete_file_id:
                vector_url = f"{KleeSettings.vector_url}{i}"
                shutil.rmtree(vector_url)

            await session.execute(delete_stmt)

            return ResponseContent(error_code=0, message=f"Refresh knowledge successfully", data={})
        except Exception as e:
            logger.error(f"Refresh knowledge error: {e}")
            return ResponseContent(error_code=-1, message=f"Refresh knowledge failed, {str(e)}", data={})

    @db_transaction
    async def import_knowledge(
            self,
            knowledge_id: str,
            file_import: LLamaFileImportRequest,
            session=None
    ):
        """
        Import knowledge from files
        Args:
            knowledge_id: Target knowledge entry ID
            file_import: File import request data
            session: Database session
        Returns:
            ResponseContent: Response indicating success/failure
        """
        try:
            if KleeSettings.local_mode is True:
                select_stmt = select(File).where(File.knowledgeId == knowledge_id)
                results = await session.execute(select_stmt)
                files = results.scalars().all()

                for file in files:
                    if os.path.exists(f"{KleeSettings.temp_file_url}{file.id}"):
                        shutil.rmtree(f"{KleeSettings.temp_file_url}{file.id}")
                    if os.path.exists(f"{KleeSettings.vector_url}{file.id}"):
                        shutil.rmtree(f"{KleeSettings.vector_url}{file.id}")

                delete_stmt = delete(File).where(File.knowledgeId == knowledge_id)
                await session.execute(delete_stmt)
                await self.llama_index_service.import_exist_dir(knowledge_id=knowledge_id, dir_path=file_import.path,
                                                                session=session)
            else:
                await self.import_exist_dir_cloud(knowledge_id=knowledge_id, dir_path=file_import.path, session=session)

            return ResponseContent(error_code=0, message="Successfully imported knowledge", data={})
        except Exception as e:
            return ResponseContent(error_code=-1, message=f"Failed to import knowledge: {str(e)}", data={})

    async def import_exist_dir_cloud(
            self,
            knowledge_id: str,
            dir_path,
            session
    ):
        """
        Import files from directory to cloud storage
        
        Args:
            knowledge_id: Target knowledge ID
            dir_path: Source directory path
            session: Database session
            
        Raises:
            Exception: If import fails
        """
        try:
            directory_path = Path(dir_path)
            files_to_add = []

            # Get knowledge data
            stmt = select(Knowledge).where(Knowledge.id == knowledge_id)
            result = await session.execute(stmt)
            knowledge_data = result.scalars().first()

            if not knowledge_data:
                raise ValueError(f"Knowledge with ID {knowledge_id} not found")

            # Process files
            for file in directory_path.rglob('*'):
                if not file.is_file() or file.name == '.DS_Store':
                    continue
                    
                file_path = str(file)
                title = file.name
                file_size = os.path.getsize(file_path)

                # Upload file to cloud
                cloud_file = await self.llama_cloud_file_service.upload_file(file_path=file_path)

                # Create file record
                knowledge_file = File(
                    id=cloud_file.id,
                    path=file_path,
                    name=title,
                    size=file_size,
                    knowledgeId=knowledge_data.id,
                    os_mtime=datetime.now().timestamp(),
                    create_at=datetime.now().timestamp(),
                    update_at=datetime.now().timestamp()
                )
                files_to_add.append(knowledge_file)

            # Bulk insert files
            if files_to_add:
                session.add_all(files_to_add)

        except Exception as e:
            logger.error(f"Failed to import directory to cloud: {str(e)}")
            raise Exception(f"Failed to import files: {str(e)}")

    @db_transaction
    async def delete_file(
            self,
            file_id: str,
            session=None
    ):
        try:
            stmt = select(File).where(File.id == file_id)
            result = await session.execute(stmt)
            file_info = result.scalars().one_or_none()
            if file_info is None:
                return ResponseContent(error_code=-1, message="File not found", data={})

            if KleeSettings.local_mode is True:
                await session.delete(file_info)

                file_id = file_info.id
                vector_url = f"{KleeSettings.vector_url}{file_id}"
                temp_url = f"{KleeSettings.temp_file_url}{file_id}"

                if os.path.exists(vector_url):
                    shutil.rmtree(vector_url)
                if os.path.exists(temp_url):
                    shutil.rmtree(temp_url)
            else:
                await session.delete(file_info)
                await self.llama_cloud_file_service.delete_file(file_id=file_id)

            return ResponseContent(error_code=0, message="Delete file successfully", data={})
        except Exception as e:
            return ResponseContent(error_code=-1, message=f"Delete file failed, {str(e)}", data={})

    async def llama_add(
            self,
            knowledge_id: str,
            file_obj: LlamaFileList
    ):
        try:
            if KleeSettings.local_mode is True:
                temp_url = f"{KleeSettings.temp_file_url}{knowledge_id}"
                if not os.path.exists(temp_url):
                    os.makedirs(temp_url, exist_ok=True)
                save_list = []
                files = file_obj.files
                for file in files:
                    file_name = ""
                    if KleeSettings.os_type == SystemTypeDiff.MAC.value:
                        file_name = file.split("/")[-1]
                    elif KleeSettings.os_type == SystemTypeDiff.WIN.value:
                        file_name = file.split("\\")[-1]

                    file_path = temp_url + "/" + file_name
                    file_id = str(uuid.uuid4())

                    lkl = LlamaKnowledge()
                    lkl.name = file_name
                    lkl.path = file_path
                    lkl.size = os.path.getsize(file)
                    lkl.file_id = file_id
                    save_list.append(lkl)

                    if not os.path.exists(f"{KleeSettings.temp_file_url}{file_id}"):
                        os.makedirs(f"{KleeSettings.temp_file_url}{file_id}", exist_ok=True)
                        await self.write_file(file, f"{KleeSettings.temp_file_url}{file_id}/{file_name}")

                    await self.llama_index_service.persist_file_to_disk_2(
                        path=f"{KleeSettings.temp_file_url}{lkl.file_id}",
                        store_dir=f"{KleeSettings.vector_url}{lkl.file_id}"
                    )
                await self.add_file_to_knowledge(
                    knowledge_id, save_list
                )
            else:
                files_path = file_obj.files
                for file_path in files_path:
                    file_name = ""
                    if KleeSettings.os_type == SystemTypeDiff.MAC.value:
                        file_name = file_path.split("/")[-1]
                    elif KleeSettings.os_type == SystemTypeDiff.WIN.value:
                        file_name = file_path.split("\\")[-1]
                    new_file = File(
                        name=file_name,
                        os_mtime=datetime.now().timestamp(),
                        format=file_name.split(".")[1],
                        size=os.path.getsize(file_path),
                        knowledgeId=knowledge_id,
                    )
                    file = await self.llama_cloud_file_service.upload_file(file_path=file_path, external_file_id=f"{knowledge_id}/{file_name}")
                    new_file.id = file.id
                    await self.save_single_file(new_file)

            return ResponseContent(error_code=0, message="Upload file successfully", data={})
        except Exception as e:
            logger.error(f"Upload file error: {e}")
            return ResponseContent(error_code=-1, message=f"Failed to upload file: {str(e)}", data={})

    @db_transaction
    async def add_file_to_knowledge(
            self,
            parent_id: str,
            file_list: list[LlamaKnowledge],
            session=None
    ):
        try:
            files = []
            for file in file_list:
                new_file = File(
                    id=file.file_id,
                    name=file.name,
                    os_mtime=datetime.now().timestamp(),
                    format=file.name.split(".")[1],
                    size=file.size,
                    knowledgeId=parent_id,
                    path=file.path
                )
                files.append(new_file)
            session.add_all(files)
        except Exception as e:
            logger.error(f"Save file to knowledge error, {e}")
            raise Exception(e)

    @db_transaction
    async def save_single_file(
            self,
            file,
            session=None
    ):
        try:
            session.add(file)
        except Exception as e:
            logger.error(f"Save single file error, {e}")
            raise Exception(e)

    def file_to_dict(
            self,
            file: File
    ) -> dict:
        return asdict(file)

    async def write_file(
            self,
            src: str,
            dest: str
    ):
        """
        Copy file from source to destination
        
        Args:
            src: Source file path
            dest: Destination file path
        """
        try:
            shutil.copy(src, dest)
        except Exception as e:
            logger.error(f"Failed to copy file from {src} to {dest}: {str(e)}")
            raise
