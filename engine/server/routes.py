"""API routes for the Apollo inference server."""

import asyncio
from uuid import UUID
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import jax.numpy as jnp

from database import crud
from database.schemas import (
    ConversationCreate,
    ConversationResponse,
    ConversationWithMessages,
    ChatCompletionRequest,
    ChatCompletionResponse,
    StreamToken,
    HealthCheck,
    MessageCreate,
)
from server.dependencies import get_database, get_model, get_tokenizer
from database.connection import engine


router = APIRouter(prefix="/v1", tags=["api"])


@router.post("/sessions", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    conversation: ConversationCreate,
    db: AsyncSession = Depends(get_database)
):
    """Create a new conversation session."""
    new_conversation = await crud.create_conversation(db, conversation)
    await db.commit()
    return new_conversation


@router.get("/sessions/{session_id}/messages", response_model=ConversationWithMessages)
async def get_session_messages(
    session_id: UUID,
    db: AsyncSession = Depends(get_database)
):
    """Get all messages for a session."""
    conversation = await crud.get_conversation_by_session_id(db, session_id)
    
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return conversation


@router.post("/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    db: AsyncSession = Depends(get_database)
):
    """
    Generate chat completion with streaming or non-streaming response.
    Automatically saves both user message and assistant response to database.
    """
    conversation = await crud.get_or_create_conversation(
        db,
        request.session_id,
        title=request.message[:50] + ("..." if len(request.message) > 50 else "")
    )
    
    user_message = await crud.create_message(
        db,
        conversation.id,
        MessageCreate(role="user", content=request.message)
    )
    await db.commit()
    
    try:
        model = get_model()
        tokenizer = get_tokenizer()
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    
    max_tokens = request.max_tokens or 512
    temperature = request.temperature or 0.8
    
    if request.stream:
        return StreamingResponse(
            stream_completion(
                model=model,
                tokenizer=tokenizer,
                conversation=conversation,
                user_message_content=request.message,
                max_tokens=max_tokens,
                temperature=temperature,
                db=db,
            ),
            media_type="text/event-stream"
        )
    else:
        response_content = await generate_completion(
            model=model,
            tokenizer=tokenizer,
            user_message=request.message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        assistant_message = await crud.create_message(
            db,
            conversation.id,
            MessageCreate(role="apollo", content=response_content)
        )
        await db.commit()
        
        return ChatCompletionResponse(
            message_id=assistant_message.id,
            role="apollo",
            content=response_content,
            created_at=assistant_message.created_at
        )


async def stream_completion(
    model,
    tokenizer,
    conversation,
    user_message_content: str,
    max_tokens: int,
    temperature: float,
    db: AsyncSession,
):
    """
    Stream tokens from the model and save complete response to database.
    
    Yields SSE (Server-Sent Events) formatted data.
    """
    import jax
    from server.dependencies import get_model as get_model_params
    import server.dependencies as deps
    
    model_structure = deps._model_structure
    params = get_model_params()
    
    input_ids = tokenizer.encode(user_message_content, add_bos=True, add_eos=False)
    current_sequence = list(input_ids)
    
    generated_tokens = []
    
    try:
        for i in range(max_tokens):
            logits = model_structure.apply(
                {'params': params},
                jnp.array([current_sequence], dtype=jnp.int32)
            )
            
            next_token_logits = logits[0, -1, :]
            
            next_token_logits = next_token_logits / temperature
            probs = jax.nn.softmax(next_token_logits, axis=-1)
            
            next_token = int(jnp.argmax(next_token_logits))
            
            if next_token == tokenizer.eos_id:
                yield f"data: {StreamToken(token='', done=True).model_dump_json()}\n\n"
                break
            
            token_text = tokenizer.decode([next_token])
            generated_tokens.append(next_token)
            current_sequence.append(next_token)
            
            yield f"data: {StreamToken(token=token_text, done=False).model_dump_json()}\n\n"
            
            await asyncio.sleep(0.01)
        
        full_response = tokenizer.decode(generated_tokens)
        assistant_message = await crud.create_message(
            db,
            conversation.id,
            MessageCreate(role="apollo", content=full_response)
        )
        await db.commit()
        
        yield f"data: {StreamToken(token='', done=True, message_id=assistant_message.id).model_dump_json()}\n\n"
        
    except Exception as e:
        yield f"data: {StreamToken(token=f'Error: {str(e)}', done=True).model_dump_json()}\n\n"
        raise


async def generate_completion(
    model,
    tokenizer,
    user_message: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate non-streaming completion."""
    import jax
    from server.dependencies import get_model as get_model_params
    import server.dependencies as deps
    
    model_structure = deps._model_structure
    params = get_model_params()
    
    input_ids = tokenizer.encode(user_message, add_bos=True, add_eos=False)
    current_sequence = list(input_ids)
    generated_tokens = []
    
    for i in range(max_tokens):
        logits = model_structure.apply(
            {'params': params},
            jnp.array([current_sequence], dtype=jnp.int32)
        )
        
        next_token_logits = logits[0, -1, :]
        next_token_logits = next_token_logits / temperature
        next_token = int(jnp.argmax(next_token_logits))
        
        if next_token == tokenizer.eos_id:
            break
        
        generated_tokens.append(next_token)
        current_sequence.append(next_token)
    
    return tokenizer.decode(generated_tokens)


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    try:
        get_model()
        get_tokenizer()
        model_loaded = True
    except RuntimeError:
        model_loaded = False
    
    return HealthCheck(
        status="healthy" if db_status == "connected" and model_loaded else "degraded",
        database=db_status,
        model_loaded=model_loaded,
    )
