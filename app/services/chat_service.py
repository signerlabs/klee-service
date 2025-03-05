import json
import os
import uuid
import re
from dataclasses import asdict
from datetime import datetime

import httpx
import nest_asyncio
import requests
import logging

from fastapi import Header, HTTPException
from sqlalchemy import select, delete
from starlette.requests import Request
from starlette.responses import StreamingResponse

from app.common.LlamaEnum import SystemTypeDiffModelType
from app.model.Chat import ChatConversation
from app.model.LlamaRequest import LlamaConversationRequest, LLamaChatRequest
from app.model.Response import ResponseContent
from app.model.base_config import BaseConfig
from app.model.knowledge import File
from app.model.note import Note
from app.services.client_sqlite_service import db_transaction

from sqlalchemy.ext.asyncio import AsyncSession
from app.services.llama_index_service import LlamaIndexService
from app.services.client_sqlite_service import async_session
from app.model.klee_settings import Settings as KleeSettings
from app.model.chat_message import ChatMessage as Llama_chat_message, Conversation as Llama_conversation, ChatMessage
from llama_index.core.base.llms.types import ChatMessage as LlmChatMessage, MessageRole


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.llama_index_service = LlamaIndexService()

    @db_transaction
    async def create_conversation_title(
            self,
            conversation_id: str,
            Authorization: str = Header(None),
            request: Request = Request,
            session=None
    ):
        """
        Create or update conversation title
        Args:
            conversation_id: ID of the conversation
            Authorization: Authorization header
            request: Request object
            session: Database session
        Returns:
            ResponseContent: Response containing conversation data
        """
        try:
            stmt = select(Llama_conversation).where(
                Llama_conversation.id == conversation_id)
            results = await session.execute(stmt)
            conversation = results.scalars().one_or_none()

            if conversation is None:
                return ResponseContent(error_code=-1, message=f"Conversation not found: {conversation_id}", data={})

            msg_stmt = select(Llama_chat_message).order_by(Llama_chat_message.create_time.asc()).limit(2).where(
                Llama_chat_message.conversation_id == conversation_id)
            results = await session.execute(msg_stmt)
            msg_list = results.scalars().all()

            if not msg_list:
                return ResponseContent(error_code=-1, message=f"Not enough messages in conversation: {conversation_id}")

            msg_content = ""
            for msg in msg_list:
                if msg.role == "user":
                    msg_content += f"Question:{msg.content}"
                else:
                    msg_content += f"Answer:{msg.content}"

            language = ""
            if conversation.language_id == "zh":
                language = "Generate title in Chinese"
            elif conversation.language_id == "en":
                language = "Generate title in English"
            elif conversation.language_id == "auto":
                language = ""

            question = """
            Generate a short and descriptive title in {language} for the following content, only in string:
            {msg_content}
            """.format(msg_content=msg_content, language=language)

            if conversation.local_mode is True:
                nest_asyncio.apply()
                if conversation.provider_id == SystemTypeDiffModelType.OLLAMA.value:
                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=json.loads(conversation.note_ids),
                        file_infos={},
                        streaming=False
                    )
                    response = query_engine.query(question)

                    conversation.title = re.sub(
                        r'<think>.*?</think>', '', response.response, flags=re.DOTALL)

                    session.add(conversation)

                    return ResponseContent(error_code=0, message="Generate title successfully", data=conversation)
                else:
                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=[],
                        file_infos={},
                        streaming=False
                    )
                    response = query_engine.query(question + language)
                    conversation.title = response.response.replace(
                        "\\n", "").strip()
                    session.add(conversation)
                    return ResponseContent(error_code=0, message="Generate title successfully", data=conversation)
            else:
                if conversation.provider_id == SystemTypeDiffModelType.OPENAI.value \
                        or conversation.provider_id == SystemTypeDiffModelType.CLAUDE.value:
                    url = f"https://xltwffswqvowersvchkj.supabase.co/functions/v1/chatService-generateConversationTitle"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {Authorization}"
                    }

                    for key, value in request.headers.items():
                        if key == "environment":
                            headers["Environment"] = value
                            break

                    os.environ["IS_TESTING"] = "True"

                    del os.environ["IS_TESTING"]
                    messages_list = [{
                        "role": "user",
                        "content": question
                    }]
                    model_name = ""

                    request_data = {
                        "provider": str(conversation.provider_id).upper(),
                        "messages": messages_list,
                        "model": model_name
                    }

                    with requests.post(url, headers=headers, json=request_data, stream=False) as response:
                        response.raise_for_status()  # 如果响应状态不是200，将引发异常
                        conversation.title = response.text
                        session.add(conversation)

                    return ResponseContent(error_code=0, message="Generate title successfully", data=conversation)
                else:
                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=json.loads(conversation.note_ids),
                        file_infos={},
                        streaming=False
                    )
                    response = query_engine.query(question)
                    conversation.title = response.response
                    session.add(conversation)

                    return ResponseContent(error_code=0, message="Generate title successfully", data=conversation)

        except Exception as e:
            logger.error(f"create_conversation_title error:{str(e)}")
            return ResponseContent(error_code=-1, message=f"Failed to generate title: {str(e)}", data={})

    @db_transaction
    async def create_conversation(
            self,
            llama_request: LlamaConversationRequest,
            session
    ):
        try:
            now_time = datetime.now().timestamp()
            chat_conversation = Llama_conversation(
                id=str(uuid.uuid4()),
                title="",
                create_time=now_time,
                is_pin=False,
                knowledge_ids=json.dumps([]),
                note_ids=json.dumps([]),
                create_at=now_time,
                update_at=now_time,
                provider_id=llama_request.provider_id,
                local_mode=llama_request.local_mode,
                language_id=llama_request.language_id,
                model_path=llama_request.model_path,
                model_name=llama_request.model_name,
                model_id=llama_request.model_id
            )

            if chat_conversation.provider_id == SystemTypeDiffModelType.OLLAMA.value \
                    or chat_conversation.provider_id == SystemTypeDiffModelType.KLEE.value \
                    or chat_conversation.provider_id == SystemTypeDiffModelType.LOCAL.value:
                chat_conversation.local_mode = True
            else:
                chat_conversation.local_mode = False

            session.add(chat_conversation)

            conversation_info = {
                "id": chat_conversation.id,
                "title": chat_conversation.title,
                "create_time": chat_conversation.create_time,
                "is_pin": chat_conversation.is_pin,
                "knowledge_ids": [],
                "note_ids": [],
                "language_id": chat_conversation.language_id,
                "model_id": chat_conversation.model_id,
                "provider_id": chat_conversation.provider_id,
                "local_mode": chat_conversation.local_mode,
                "system_prompt": chat_conversation.model_path,
                "model_path": chat_conversation.model_path,
                "model_name": chat_conversation.model_name,
                "create_at": chat_conversation.create_at,
                "update_at": chat_conversation.update_at
            }

            return ResponseContent(error_code=0, message="", data={"conversation": conversation_info})
        except Exception as e:
            logger.error(f"create_conversation error:{str(e)}")
            raise Exception(f"{e}")

    @db_transaction
    async def delete_conversation(
            self,
            session,
            conversation_id: str
    ):
        """
        Delete a conversation and its messages
        Args:
            session: Database session
            conversation_id: ID of the conversation to delete
        Returns:
            ResponseContent: Response indicating success/failure
        """
        try:
            result = await session.execute(select(Llama_conversation).filter(Llama_conversation.id == conversation_id))
            conversation = result.scalar_one_or_none()
            if conversation is None:
                raise HTTPException(
                    status_code=404, detail="Conversation not found")

            await session.delete(conversation)
            return ResponseContent(error_code=0, message="Successfully deleted conversation", data=None)
        except Exception as e:
            logger.error(f"delete_conversation error: {str(e)}")
            raise Exception(f"Failed to delete conversation: {str(e)}")

    @db_transaction
    async def delete_message(
            self,
            session,
            message_id: str
    ):
        """
        Delete a message and its related messages
        Args:
            session: Database session
            message_id: ID of the message to delete
        Returns:
            ResponseContent: Response indicating success/failure
        """
        try:
            stmt = select(Llama_chat_message).where(
                Llama_chat_message.id == message_id)
            result = await session.execute(stmt)
            message = result.scalar_one_or_none()
            if message is None:
                raise HTTPException(
                    status_code=404, detail="Message not found")

            create_time = message.create_time
            stmt = (select(Llama_chat_message).where(Llama_chat_message.create_time == create_time)
                    .order_by(Llama_chat_message.create_time.desc())).limit(2)
            result = await session.execute(stmt)
            new_messages = result.scalars().all()

            delete_ids = []
            for msg in new_messages:
                del_stmt = delete(Llama_chat_message).where(
                    Llama_chat_message.id == msg.id)
                await session.execute(del_stmt)
                delete_ids.append(msg.id)
            await session.commit()

            return ResponseContent(error_code=0, message="Successfully deleted message", data={"ids": delete_ids})
        except Exception as e:
            await session.rollback()
            logger.error(f"delete_message error: {str(e)}")
            return ResponseContent(error_code=-1, message=f"Failed to delete message: {str(e)}", data={})

    @db_transaction
    async def update_conversation(
            self,
            session,
            conversation_id: str,
            request: LlamaConversationRequest
    ):
        try:
            result = await session.execute(select(Llama_conversation).filter(Llama_conversation.id == conversation_id))
            conversation = result.scalar_one_or_none()
            if conversation is None:
                raise HTTPException(
                    status_code=404, detail="Conversation not found")

            conversation.title = request.title
            conversation.is_pin = request.is_pin
            conversation.update_at = datetime.now().timestamp()

            await session.commit()
            await session.refresh(conversation)

            conversation_response = {
                "id": conversation_id,
                "title": conversation.title,
                "create_time": conversation.create_time,
                "is_pin": conversation.is_pin,
                "language_id": conversation.language_id,
                "create_at": conversation.create_at,
                "update_at": conversation.update_at
            }
            return ResponseContent(error_code=0, message="Update conversational information successfully", data=conversation_response)
        except Exception as e:
            await session.rollback()
            logger.error(f"update_conversation error:{str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Update conversational information failed: {str(e)}")

    @db_transaction
    async def llama_get_conversation_message(
            self,
            conversation_id: str = None,
            session=None
    ):
        try:
            stmt = select(Llama_conversation).where(
                Llama_conversation.id == conversation_id)
            result = await session.execute(stmt)
            conversation = result.scalars().first()

            if conversation is None:
                return ResponseContent(error_code=-1, message="conversation not found", data={})

            if conversation.provider_id == SystemTypeDiffModelType.OLLAMA.value:
                os.system(f"ollama stop {conversation.model_id}")

            stmt = select(Llama_chat_message).where(
                Llama_chat_message.conversation_id == conversation_id)
            result_messages = await session.execute(stmt)
            message_list = result_messages.scalars().all()

            conversation_info = {
                "id": conversation.id,
                "knowledge_ids": json.loads(conversation.knowledge_ids),
                "note_ids": json.loads(conversation.note_ids),
                "title": conversation.title,
                "is_pin": conversation.is_pin,
                "create_time": conversation.create_time,
                "local_mode": conversation.local_mode,
                "provider_id": conversation.provider_id,
                "model_id": conversation.model_id,
                "language_id": conversation.language_id,
                "system_prompt": conversation.system_prompt,
                "model_path": conversation.model_path,
                "create_at": conversation.create_at,
                "update_at": conversation.update_at
            }

            message_infos = []
            if len(message_list) > 0:
                message_infos = [
                    {
                        "id": message.id,
                        "role": message.role,
                        "content": message.content,
                        "create_time": message.create_time,
                        "status": message.status,
                        "error_message": message.error_message,
                        "create_at": message.create_at,
                        "update_at": message.update_at,
                    }
                    for message in message_list
                ]
            return ResponseContent(error_code=0, message="",
                                   data={"conversation": conversation_info, "messages": message_infos})

        except Exception as e:
            logger.error(f"get_conversation_message error:{str(e)}")
            raise Exception("Get conversation message failed")

    @db_transaction
    async def get_all_chat_conversations(
            self,
            session,
            keyword: str = None
    ):
        try:
            stmt = select(Llama_conversation).filter(Llama_conversation.title.like(f"%{keyword}%")).order_by(
                Llama_conversation.create_time.desc())
            result = await session.execute(stmt)
            conversations = result.scalars().all()

            if len(conversations) < 0:
                return ResponseContent(error_code=0, message="Get all conversations successfully", data=[])

            conversations_list = [
                {
                    "id": cv.id,
                    "title": cv.title,
                    "create_time": cv.create_time,
                    "create_at": cv.create_at,
                    "update_at": cv.update_at,
                    "is_pin": cv.is_pin,
                    "knowledge_ids": json.loads(cv.knowledge_ids),
                    "note_ids": json.loads(cv.note_ids)
                }
                for cv in conversations
            ]

            return ResponseContent(error_code=0, message="Get all conversations successfully", data=conversations_list)
        except Exception as e:
            logger.error(f"Get all conversations failed: {str(e)}")
            return ResponseContent(error_code=1, message=f"Get all conversations failed: {str(e)}", data=None)

    @db_transaction
    async def get_conversation_detail(
            self,
            session,
            conversation_id: str
    ):
        try:
            async with async_session() as session:
                # 查询指定的 ChatConversation 记录
                stmt = select(ChatConversation).where(
                    ChatConversation.id == conversation_id)
                result = await session.execute(stmt)
                conversation = result.scalar_one_or_none()

                if not conversation:
                    return ResponseContent(error_code=1, message="Conversation not found", data=None)

                if conversation.provider_id == SystemTypeDiffModelType.OLLAMA.value:
                    os.system(f"ollama stop {conversation.model_id}")

                # 构造返回的数据结构
                conversation_detail = {
                    "id": conversation.id,
                    "title": conversation.title,
                    "latestMessage": conversation.latestMessage,
                    "knowledgeIds": conversation.knowledgeIds,
                    "noteIds": conversation.noteIds,
                    "fileIds": conversation.fileIds,
                    "createAt": conversation.createAt,
                    "updateAt": conversation.updateAt,
                    "systemPrompt": conversation.systemPrompt,
                    "replyLanguage": conversation.replyLanguage,
                    "llmId": conversation.llmId,
                    "knowledgeContent": conversation.knowledgeContent
                }

                return ResponseContent(error_code=0, message="Get conversation detail successfully", data=conversation_detail)
        except Exception as e:
            logger.error(f"Get conversation detail failed: {str(e)}")
            return ResponseContent(error_code=1, message=f"Get conversation detail failed: {str(e)}", data=None)

    @db_transaction
    async def get_conversation_messages(
            self,
            session,
            conversation_id: str
    ):
        """
        待定处理
        """
        try:
            # 查询指定对话的所有消息
            stmt = select(ChatMessage).where(ChatMessage.conversation_id == conversation_id).order_by(
                ChatMessage.create_at)
            result = await session.execute(stmt)
            messages = result.scalars().all()

            # 将查询结果转换为字典列表
            messages_list = [
                {
                    "id": msg.id,
                    "conversationId": msg.conversation_id,
                    "question": msg.question,
                    "answer": msg.answer,
                    "status": msg.status,
                    "createAt": msg.create_at,
                    "updateAt": msg.update_at
                }
                for msg in messages
            ]

            return ResponseContent(error_code=0, message="Get all conversation messages successfully", data=messages_list)
        except Exception as e:
            logger.error(f"Get all conversation messages failed: {str(e)}")
            return ResponseContent(error_code=1, message=f"Get all conversation messages failed: {str(e)}", data=None)

    @db_transaction
    async def rot_chat(
            self,
            chat_request: LLamaChatRequest,
            Authorization: str = Header(None),
            request: Request = Request,
            session=None
    ):
        try:
            stmt = select(Llama_conversation).where(
                Llama_conversation.id == chat_request.conversation_id)
            result = await session.execute(stmt)
            a_conversation = result.scalar_one_or_none()

            stmt = select(Llama_chat_message).where(
                Llama_chat_message.conversation_id == chat_request.conversation_id).order_by(
                Llama_chat_message.create_time.desc()).limit(4)
            results = await session.execute(stmt)

            provider_id = a_conversation.provider_id
            model_id = a_conversation.model_id
            model_path = a_conversation.model_path

            # 释放内存
            if KleeSettings.un_load is True:
                # try_release()
                self.llama_index_service.release_memory()

                if a_conversation.provider_id is None or a_conversation.provider_id == "":
                    return ResponseContent(error_code=-1, message="Please select a chatbot model", data={})

                api_type = None
                if KleeSettings.local_mode is True:
                    if provider_id == SystemTypeDiffModelType.OLLAMA.value and model_id is not None and not model_id == "":
                        await self.llama_index_service.load_llm(provider_id=provider_id, model_name=model_id)
                    elif provider_id == SystemTypeDiffModelType.KLEE.value and model_id is not None and not model_id == "":
                        model_path = f"{KleeSettings.llm_path}{str(model_id).lower()}.gguf"
                        await self.llama_index_service.load_llm(provider_id=provider_id, model_name=model_path)
                    elif provider_id == SystemTypeDiffModelType.LOCAL.value:
                        await self.llama_index_service.load_llm(provider_id=provider_id, model_name=model_path)
                else:
                    api_base_url = None
                    if not provider_id == SystemTypeDiffModelType.OPENAI.value and not provider_id == SystemTypeDiffModelType.CLAUDE.value:
                        diy_stmt = select(BaseConfig).where(
                            BaseConfig.id == provider_id)
                        result = await session.execute(diy_stmt)
                        config_data = result.scalars().one_or_none()
                        if not str(a_conversation.model_name).find("claude") == -1:
                            os.environ["ANTHROPIC_API_KEY"] = config_data.apiKey
                            api_type = "anthropic"
                            model_id = a_conversation.model_name
                        elif not str(a_conversation.model_name).find("gpt") == -1:
                            os.environ["OPENAI_API_KEY"] = config_data.apiKey
                            api_type = "openai"
                            model_id = a_conversation.model_name
                        elif not str(a_conversation.model_name).find("deepseek") == -1:
                            os.environ["DEEPSEEK_API_KEY"] = config_data.apiKey
                            api_type = "deepseek"
                            model_id = a_conversation.model_name

                            if config_data.baseUrl.find("luchentech") != -1:
                                api_base_url = "https://cloud.luchentech.com/api/maas"

                    await self.llama_index_service.load_llm(provider_id=provider_id, model_name=model_id, api_type=api_type,
                                                            api_base_url=api_base_url)

            chat_messages = results.scalars().all()

            question = chat_request.question
            language = ""
            if a_conversation.language_id == "zh":
                language = f".Please answer in Chinese."
            elif a_conversation.language_id == "en":
                language = f".Please reply in English."
            elif a_conversation.language_id == "auto":
                language = ""

            file_infos = {}
            knowledge_list = []
            knowledge_ids = json.loads(a_conversation.knowledge_ids)
            if len(knowledge_ids) > 0:
                for knowledge_id in knowledge_ids:
                    stmt = select(File).where(File.knowledgeId == knowledge_id)
                    result = await session.execute(stmt)
                    files = result.scalars().all()

                    if len(files) > 0:
                        file_infos[knowledge_id] = files
                        knowledge_list.append(files)

            note_list = []
            # 这里加入查找note笔记的内容
            note_ids = json.loads(a_conversation.note_ids)
            if len(note_ids) > 0:
                for note_id in note_ids:
                    stmt = select(Note).where(Note.id == note_id)
                    result = await session.execute(stmt)
                    note = result.scalar_one_or_none()
                    if note is not None:
                        note_list.append(note)

            nest_asyncio.apply()

            # 本地模式
            if a_conversation.local_mode is True:
                if a_conversation.provider_id == SystemTypeDiffModelType.OLLAMA.value:
                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=json.loads(a_conversation.note_ids),
                        file_infos=file_infos
                    )

                    response = query_engine.query(question + language)

                    return StreamingResponse(
                        self.generate_data(
                            session=session,
                            response=response,
                            question=question,
                            conversation_id=chat_request.conversation_id),
                        media_type="text/event-stream"
                    )
                else:
                    history_message = "|||"

                    chat_history = []
                    for item in chat_messages:
                        role = None
                        if item.role == "user":
                            role = MessageRole.USER
                        elif item.role == "assistant":
                            role = MessageRole.ASSISTANT
                        chat_message = LlmChatMessage(
                            role=role,
                            content=item.content
                        )
                        chat_history.append(chat_message)
                        history_message = history_message + item.role + ":" + item.content + "|||"

                        chat_history.append(LlmChatMessage(
                            role=MessageRole.USER,
                            content=question + language
                        ))

                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=json.loads(a_conversation.note_ids),
                        file_infos=file_infos
                    )

                    real_question = question + language

                    real_question += """.If the answer is unrelated to the question, you can freely express yourself. \n"
                                   f".Do not directly output the provided text content. \n"
                                   f".If no text is provided, please provide your own response and organize the answer. \n"""

                    response = query_engine.query(real_question)
                    response_coroutine = self.generate_data(
                        response=response, question=question, conversation_id=chat_request.conversation_id)
                    return StreamingResponse(response_coroutine,
                                             media_type="text/event-stream")
            else:
                if a_conversation.provider_id == SystemTypeDiffModelType.OPENAI.value \
                        or a_conversation.provider_id == SystemTypeDiffModelType.CLAUDE.value \
                        or a_conversation.provider_id == SystemTypeDiffModelType.DEEPSEEK.value:
                    url = f"https://xltwffswqvowersvchkj.supabase.co/functions/v1/chatService-createCompletion"
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {Authorization}"
                    }

                    for key, value in request.headers.items():
                        if key.lower() == "environment":
                            headers["Environment"] = value
                            break

                    nest_asyncio.apply()
                    os.environ["IS_TESTING"] = "True"
                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=json.loads(a_conversation.note_ids),
                        file_infos=file_infos
                    )

                    content = await self.llama_index_service.get_retrieve_notes_content(question=question + language,
                                                                                        query_engine=query_engine)
                    context = content

                    del os.environ["IS_TESTING"]

                    messages_list = []
                    for item in chat_messages:
                        role = None
                        if item.role == "user":
                            role = MessageRole.USER
                        elif item.role == "assistant":
                            role = MessageRole.ASSISTANT
                        chat_history_msg = {
                            "role": role.value,
                            "content": item.content
                        }
                        messages_list.append(chat_history_msg)

                    messages_list.append({
                        "role": "user",
                        "content": f"1、{language}，using as many sentences as possible from the text I provided: "
                        f"```{context}``` to support the answer"
                        f"2、If the answer is unrelated to the question, you can freely express yourself"
                        f"3、Do not directly output the provided text content"
                        f"4、If no text is provided, please provide your own response and organize the answer."
                        f"My question is：{question}"
                    })

                    request_data = {
                        "provider": str(a_conversation.provider_id).upper(),
                        "messages": messages_list,
                        "model": a_conversation.model_name
                    }

                    return StreamingResponse(
                        self.generate_data_2(url, headers, request_data, conversation_id=a_conversation.id,
                                             question=question),
                        media_type="text/event-stream")
                else:
                    query_engine = await self.llama_index_service.combine_query(
                        note_ids=json.loads(a_conversation.note_ids),
                        file_infos=file_infos
                    )
                    response = query_engine.query(question + language)
                    return StreamingResponse(self.generate_data(response=response, question=question, conversation_id=chat_request.conversation_id),
                                             media_type="text/event-stream")
        except requests.RequestException as e:
            logger.error(f"Error: {e.response.status_code}- {e.response.text}")

    async def generate_data(
            self,
            session,
            response,
            question: str,
            conversation_id: str = None
    ):
        """
        Generate streaming response data for chat messages

        Args:
            session: Database session
            response: Response from LLM
            question: User's question
            conversation_id: ID of the conversation

        Yields:
            Server-sent events containing chat message data
        """
        try:
            message_id = str(uuid.uuid4())
            create_at = datetime.now().timestamp()

            # Create user message
            user_message = Llama_chat_message(
                id=message_id,
                role="user",
                content=question,
                create_time=create_at,
                modify_time=create_at,
                create_at=create_at,
                update_at=create_at,
                conversation_id=conversation_id,
                status="success",
                error_message="success"
            )

            # Create assistant message
            rot_message = Llama_chat_message(
                id=str(uuid.uuid4()),
                role="assistant",
                content="",
                modify_time=create_at,
                create_time=create_at,
                create_at=create_at,
                update_at=create_at,
                conversation_id=conversation_id,
                status="pending",
                error_message=""
            )

            # Convert messages to dict format
            message_json_obj = asdict(user_message)
            message_json_rob_obj = asdict(rot_message)

            # Save messages to database
            session.add(user_message)
            session.add(rot_message)
            await session.flush()
            await session.commit()

            # Send initial event
            yield "event: sending\n"
            yield f"data: {json.dumps({'userMessage': message_json_obj, 'botMessage': message_json_rob_obj, 'conversation_id': conversation_id})}\n\n"

            # Process streaming response
            async def process_stream(stream):
                async for item in stream:
                    rot_message.content += item
                    message_json_rob_obj['content'] += item
                    yield "event: pending\n"
                    yield f"data: {json.dumps(message_json_rob_obj)}\n\n"

            # Handle both async and sync iterators
            if hasattr(response.response_gen, '__aiter__'):
                async for data in process_stream(response.response_gen):
                    yield data
            else:
                for item in response.response_gen:
                    rot_message.content += item
                    message_json_rob_obj['content'] += item
                    yield "event: pending\n"
                    yield f"data: {json.dumps(message_json_rob_obj)}\n\n"

            # Update message status on completion
            rot_message.status = "success"
            async with async_session() as new_session:
                stmt = select(Llama_chat_message).where(
                    Llama_chat_message.id == rot_message.id)
                result = await new_session.execute(stmt)
                db_rot_message = result.scalar_one()
                db_rot_message.content = rot_message.content
                db_rot_message.status = "success"
                await new_session.commit()

            # Send success event
            message_json_rob_obj['status'] = "success"
            yield "event: success\n"
            yield f"data: {json.dumps({'userMessage': message_json_obj, 'botMessage': message_json_rob_obj, 'conversation_id': conversation_id})}\n\n"

        except Exception as e:
            logger.error(f"Error generating chat response: {str(e)}")
            # Handle error case
            async with async_session() as error_session:
                async with error_session.begin():
                    stmt = select(Llama_chat_message).where(
                        Llama_chat_message.id == rot_message.id)
                    result = await error_session.execute(stmt)
                    error_message = result.scalar_one()
                    error_message.status = "error"
                    error_message.error_code = "chat_error"
                    error_message.error_message = str(e)
                    await error_session.commit()

            message_json_rob_obj.update({
                "content": rot_message.content,
                "status": "error",
                "error_code": "chat_error",
                "error_message": str(e)
            })

            yield "event: error\n"
            yield f"data: {json.dumps({'userMessage': message_json_obj, 'botMessage': message_json_rob_obj, 'conversation_id': conversation_id})}\n\n"

    async def generate_data_2(
            self,
            url: str,
            headers,
            request_data,
            question: str,
            conversation_id: str = None):

        session = get_db_session()

        message_id = str(uuid.uuid4())

        create_at = datetime.now().timestamp()

        user_message = Llama_chat_message(
            id=message_id,
            role="user",
            content=question,
            create_time=create_at,
            modify_time=datetime.now().timestamp(),
            create_at=datetime.now().timestamp(),
            conversation_id=conversation_id,
            status="success",
            error_message="success"
        )
        rot_message = Llama_chat_message(
            id=str(uuid.uuid4()),
            role="assistant",
            content="",
            modify_time=datetime.now().timestamp(),
            create_time=create_at,
            create_at=datetime.now().timestamp(),
            conversation_id=conversation_id,
            # parent_id=message_id,
            status="pending",
            error_message=""
        )

        message_json_obj = {
            "id": user_message.id,
            "role": user_message.role,
            "content": user_message.content,
            "create_time": user_message.create_time,
            "create_at": user_message.create_at,
            "conversation_id": user_message.conversation_id,
            "status": user_message.status,
            "error_message": user_message.error_message
        }

        message_json_rob_obj = {
            "id": rot_message.id,
            "role": rot_message.role,
            "content": rot_message.content,
            "create_time": rot_message.create_time,
            "create_at": rot_message.create_at,
            "conversation_id": rot_message.conversation_id,
            "status": rot_message.status,
            "error_message": rot_message.error_message
        }

        message_list_json = []
        try:
            message_list_json = []

            session.add(user_message)
            session.add(rot_message)

            message_list_json.append(message_json_obj)
            yield "event: sending\n"
            yield "data: {\"userMessage\":" + json.dumps(
                message_json_obj) + ", \"botMessage\": " + json.dumps(
                message_json_rob_obj) + ", \"conversation_id\": \"" + conversation_id + "\" }\n\n"

            async with httpx.AsyncClient() as client:
                async with client.stream(method="POST", url=url, headers=headers, json=request_data) as response:

                    if response.status_code != 200:
                        error_content = ""
                        async for item in response.aiter_bytes():
                            error_content += item.decode(encoding="utf-8")

                        error_content = json.loads(error_content)
                        message_json_rob_obj["content"] = rot_message.content
                        message_json_rob_obj['status'] = "error"
                        message_json_rob_obj['error_code'] = error_content['code']
                        message_json_rob_obj['error_message'] = error_content['code']

                        rot_message.status = "error"
                        rot_message.error_code = error_content['code']
                        rot_message.error_message = error_content['code']

                        yield "event: error\n"
                        yield "data: {\"userMessage\":" + json.dumps(
                            message_json_obj) + ", \"botMessage\": " + json.dumps(
                            message_json_rob_obj) + ", \"conversation_id\": \"" + conversation_id + "\" }\n\n"

                        session.add(user_message)
                        session.add(rot_message)

                        await session.commit()
                    else:
                        async for item in response.aiter_bytes():
                            rot_message.content += item.decode(
                                encoding="utf-8")
                            message_json_rob_obj['content'] += item.decode(
                                encoding="utf-8")
                            data = "data: " + \
                                json.dumps(message_json_rob_obj) + "\n\n"
                            yield "event: pending\n"
                            yield f"{data}"
                        rot_message.status = "success"
                        message_json_rob_obj['status'] = "success"
                        message_list_json.append(message_json_rob_obj)
                        message_json_rob_obj["content"] = rot_message.content

                        yield "event: success\n"
                        yield "data: {\"userMessage\":" + json.dumps(
                            message_json_obj) + ", \"botMessage\": " + json.dumps(
                            message_json_rob_obj) + ", \"conversation_id\": \"" + conversation_id + "\" }\n\n"
                        await session.commit()
        except Exception as e:
            rot_message.status = "error"

            rot_message.error_code = "chat_error"
            rot_message.error_message = str(e)
            message_json_rob_obj["content"] = rot_message.content
            message_json_rob_obj['status'] = "error"
            message_json_rob_obj['error_code'] = "chat_error"
            message_json_rob_obj['error_message'] = "chat_error"

            yield "event: error\n"
            yield "data: {\"userMessage\":" + json.dumps(
                message_json_obj) + ", \"botMessage\": " + json.dumps(
                message_json_rob_obj) + ", \"conversation_id\": \"" + conversation_id + "\" }\n\n"

            session.add(user_message)
            session.add(rot_message)

            await session.commit()


def get_db_session() -> AsyncSession:
    """
    Get a new db session
    Returns:
        AsyncSession: A new db session
    """
    return async_session()
