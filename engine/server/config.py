"""
Server configuration using Pydantic Settings.

Create a .env file in the engine/ directory with:
    DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/apollo_db
    HOST=0.0.0.0
    PORT=8000
    MODEL_PATH=./models/gemma/gemma-3-270m
    TOKENIZER_PATH=./models/gemma/gemma-3-270m/tokenizer.model
    ALLOWED_ORIGINS=http://localhost:3000
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    database_url: str = "sqlite+aiosqlite:///./apollo.db"
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    model_path: str = "./models/gemma/gemma-3-270m"
    tokenizer_path: str = "./models/gemma/gemma-3-270m/tokenizer.model"
    
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    default_max_tokens: int = 512
    default_temperature: float = 0.8
    
    allowed_origins: List[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    enable_metrics: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
