"""CRUD operations for database models."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from typing import Optional, List
from uuid import UUID, uuid4

from database.models import Conversation, Message, MessageRole
from database.schemas import ConversationCreate, MessageCreate


async def get_conversation_by_session_id(
    db: AsyncSession,
    session_id: UUID
) -> Optional[Conversation]:
    """Get conversation by session_id."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.session_id == session_id)
        .options(selectinload(Conversation.messages))
    )
    return result.scalar_one_or_none()


async def get_conversation_by_id(
    db: AsyncSession,
    conversation_id: UUID
) -> Optional[Conversation]:
    """Get conversation by id."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .options(selectinload(Conversation.messages))
    )
    return result.scalar_one_or_none()


async def create_conversation(
    db: AsyncSession,
    conversation: ConversationCreate,
    session_id: Optional[UUID] = None
) -> Conversation:
    """Create a new conversation."""
    if session_id is None:
        session_id = uuid4()
    
    db_conversation = Conversation(
        session_id=session_id,
        title=conversation.title,
    )
    db.add(db_conversation)
    await db.flush()
    await db.refresh(db_conversation)
    return db_conversation


async def update_conversation_title(
    db: AsyncSession,
    conversation_id: UUID,
    title: str
) -> Optional[Conversation]:
    """Update conversation title."""
    result = await db.execute(
        select(Conversation).where(Conversation.id == conversation_id)
    )
    conversation = result.scalar_one_or_none()
    
    if conversation:
        conversation.title = title
        await db.flush()
        await db.refresh(conversation)
    
    return conversation


async def create_message(
    db: AsyncSession,
    conversation_id: UUID,
    message: MessageCreate
) -> Message:
    """Create a new message in a conversation."""
    db_message = Message(
        conversation_id=conversation_id,
        role=MessageRole(message.role),
        content=message.content,
    )
    db.add(db_message)
    await db.flush()
    await db.refresh(db_message)
    return db_message


async def get_messages_by_conversation(
    db: AsyncSession,
    conversation_id: UUID
) -> List[Message]:
    """Get all messages for a conversation."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    return list(result.scalars().all())


async def get_or_create_conversation(
    db: AsyncSession,
    session_id: UUID,
    title: str = "New Chat"
) -> Conversation:
    """Get existing conversation or create new one."""
    conversation = await get_conversation_by_session_id(db, session_id)
    
    if conversation is None:
        conversation = await create_conversation(
            db,
            ConversationCreate(title=title),
            session_id=session_id
        )
    
    return conversation
