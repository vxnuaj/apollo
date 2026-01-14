"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import get_db
from server.config import Settings, settings


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return settings


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Database session dependency."""
    async for session in get_db():
        yield session


_model_instance = None
_model_structure = None
_tokenizer_instance = None


def set_model(model):
    """Set the global model instance."""
    global _model_instance
    _model_instance = model


def get_model():
    """Get the global model instance."""
    if _model_instance is None:
        raise RuntimeError("Model not loaded. Call set_model() first.")
    return _model_instance


def set_tokenizer(tokenizer):
    """Set the global tokenizer instance."""
    global _tokenizer_instance
    _tokenizer_instance = tokenizer


def get_tokenizer():
    """Get the global tokenizer instance."""
    if _tokenizer_instance is None:
        raise RuntimeError("Tokenizer not loaded. Call set_tokenizer() first.")
    return _tokenizer_instance
