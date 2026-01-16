"""Pydantic schemas for API request/response validation."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from uuid import UUID
from datetime import datetime


# Message Schemas
class MessageBase(BaseModel):
    """Base message schema."""
    role: str = Field(..., pattern="^(user|apollo)$")
    content: str = Field(..., min_length=1)


class MessageCreate(MessageBase):
    """Schema for creating a message."""
    pass


class MessageResponse(MessageBase):
    """Schema for message response."""
    id: UUID
    conversation_id: UUID
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Conversation Schemas
class ConversationBase(BaseModel):
    """Base conversation schema."""
    title: str = Field(default="New Chat", max_length=500)


class ConversationCreate(ConversationBase):
    """Schema for creating a conversation."""
    pass


class ConversationResponse(ConversationBase):
    """Schema for conversation response."""
    id: UUID
    session_id: UUID
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ConversationWithMessages(ConversationResponse):
    """Schema for conversation with all messages."""
    messages: List[MessageResponse] = []
    
    model_config = ConfigDict(from_attributes=True)


# Chat Completion Schemas
class ChatCompletionRequest(BaseModel):
    """Schema for chat completion request."""
    session_id: UUID
    message: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="gemma-3-270m")
    stream: bool = Field(default=True)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class StreamToken(BaseModel):
    """Schema for streaming token response."""
    token: str
    done: bool = False
    message_id: Optional[UUID] = None


class ChatCompletionResponse(BaseModel):
    """Schema for non-streaming chat completion response."""
    message_id: UUID
    role: str = "apollo"
    content: str
    created_at: datetime


# Health Check Schema
class HealthCheck(BaseModel):
    """Schema for health check response."""
    status: str
    database: str
    model_loaded: bool
    version: str = "0.1.0"
