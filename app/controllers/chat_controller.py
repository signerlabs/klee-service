from fastapi import APIRouter, Depends, Header, Request

from langchain_community.chat_message_histories import ChatMessageHistory
import logging
from app.services.chat_service import ChatService
from app.model.LlamaRequest import LLamaChatRequest, LlamaConversationRequest

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

history = ChatMessageHistory()
router = APIRouter()

cancel_events = {}


class ChatController:
    def __init__(self):
        self.chat_service = ChatService()

    async def create_conversation_title(
            self,
            conversation_id: str,
            Authorization: str = Header(None),
            request: Request = Request
    ):
        """
        Create or update a conversation title
        Args:
            conversation_id: ID of the conversation
            Authorization: Authorization header
            request: Request object
        Returns:
            Response containing updated conversation data
        """
        return await self.chat_service.create_conversation_title(
            conversation_id=conversation_id,
            Authorization=Authorization,
            request=request
        )

    async def create_conversation(
            self,
            llama_request: LlamaConversationRequest
    ):
        """
        Create a new conversation
        Args:
            llama_request: Conversation creation request data
        Returns:
            Response containing created conversation data
        """
        return await self.chat_service.create_conversation(llama_request=llama_request)

    async def delete_conversation(
            self,
            conversation_id: str
    ):
        """
        Delete a conversation
        Args:
            conversation_id: ID of the conversation to delete
        Returns:
            Response indicating success/failure
        """
        return await self.chat_service.delete_conversation(conversation_id=conversation_id)

    async def delete_message(
            self,
            message_id: str,
    ):
        """
        Delete a message
        Args:
            message_id: ID of the message to delete
        Returns:
            Response indicating success/failure
        """
        return await self.chat_service.delete_message(message_id=message_id)

    async def update_conversation(
            self,
            conversation_id: str,
            request: LlamaConversationRequest
    ):
        """
        Update conversation details
        Args:
            conversation_id: ID of the conversation to update
            request: Updated conversation data
        Returns:
            Response containing updated conversation data
        """
        return await self.chat_service.update_conversation(conversation_id=conversation_id, request=request)

    async def llama_get_conversation_message(
            self,
            conversation_id: str = None
    ):
        return await self.chat_service.llama_get_conversation_message(conversation_id=conversation_id)

    async def get_conversation_detail(
            self,
            conversation_id: str
    ):
        return await self.chat_service.get_conversation_detail(conversation_id=conversation_id)

    async def get_conversation_messages(
            self,
            conversation_id: str
    ):
        return await self.chat_service.get_conversation_messages(conversation_id=conversation_id)

    async def get_all_chat_conversations(
            self,
            keyword: str = None
    ):
        return await self.chat_service.get_all_chat_conversations(keyword=keyword)

    async def rot_chat(
            self,
            chat_request: LLamaChatRequest,
            Authorization: str = Header(None),
            request: Request = Request
    ):
        return await self.chat_service.rot_chat(chat_request=chat_request, Authorization=Authorization, request=request)


def class_to_dict(obj):
    if hasattr(obj, "__dict__"):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = class_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [class_to_dict(item) for item in obj]
    else:
        return obj


@router.post("/conversation/title/{conversation_id}")
async def create_conversation_title(
        conversation_id: str,
        Authorization: str = Header(None),
        request: Request = Request,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.create_conversation_title(conversation_id=conversation_id,
                                                                   Authorization=Authorization, request=request)


@router.post("/conversation")
async def create_conversation(
        llama_request: LlamaConversationRequest,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.create_conversation(llama_request=llama_request)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
        conversation_id: str,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.delete_conversation(conversation_id=conversation_id)


@router.delete("/message/{message_id}")
async def delete_message(
        message_id: str,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.delete_message(message_id=message_id)


@router.put("/conversations/{conversation_id}")
async def update_conversation(
        conversation_id: str,
        request: LlamaConversationRequest,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.update_conversation(conversation_id=conversation_id, request=request)


@router.get("/conversations/{conversation_id}")
async def llama_get_conversation_message(
        conversation_id: str = None,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.llama_get_conversation_message(conversation_id=conversation_id)


@router.get('/conversations')
async def get_all_chat_conversations(
        keyword: str = None,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.get_all_chat_conversations(keyword=keyword)


# 获取指定的Conversation的详情
@router.get('/conversations/{conversation_id}')
async def get_conversation_detail(
        conversation_id: str,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.get_conversation_detail(conversation_id=conversation_id)


# 获取指定Conversation的聊天记录
@router.get('/conversations/{conversation_id}/messages')
async def get_conversation_messages(
        conversation_id: str,
        conversation_controller: ChatController = Depends(ChatController)
):
    """
    Retrieve all messages for a specific conversation
    Args:
        conversation_id: ID of the conversation
        conversation_controller: ChatController instance
    Returns:
        Response containing conversation messages
    """
    return await conversation_controller.get_conversation_messages(conversation_id=conversation_id)


@router.post("/rot/chat")
async def rot_chat(
        chat_request: LLamaChatRequest,
        Authorization: str = Header(None),
        request: Request = Request,
        conversation_controller: ChatController = Depends(ChatController)
):
    return await conversation_controller.rot_chat(chat_request=chat_request, Authorization=Authorization, request=request)


