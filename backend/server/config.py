from dotenv import load_dotenv
from pydantic import model_validator, field_validator
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import sys
import os

file_dir_path, _ = os.path.split(os.path.abspath(__file__))
sys.path.insert(0, file_dir_path)
from models.model_registry import MODEL_REGISTRY

load_dotenv()

def get_model_name_param(model_path: Path):
    root_dir_name, model_name_param = os.path.split(str(model_path))
    _, model_name = os.path.split(root_dir_name)
    return model_name, model_name_param

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./apollo.db"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    model_path: Path = Path("gemma/gemma-3-270m")
    tokenizer_path: Path = Path("gemma/gemma-3-270m/tokenizer.model")
    max_batch_size: int = 32
    max_sequence_tok_length: int = 32_000
    temperature: float = 0.8
    allowed_origins: List[str] = [] 

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def validate_allowed_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def validate_model_name(self):
        model_name, model_name_param = get_model_name_param(self.model_path)
        assert model_name in MODEL_REGISTRY, ValueError(f"{model_name} is not supported, not in MODEL_CONTEXT_MAPPING")
        assert model_name_param in MODEL_REGISTRY[model_name], ValueError(f"{model_name} is not supported. Only {','.join(MODEL_REGISTRY[model_name].keys())} are supported")
        return self

    @model_validator(mode="after")
    def check_max_toks(self):
        model_name, model_name_param = get_model_name_param(self.model_path)
        max_context_length = MODEL_REGISTRY[model_name][model_name_param]["context_len"]
        if self.max_sequence_tok_length > max_context_length:
            raise ValueError(f'Model {model_name_param} only supports a maximum context length of {max_context_length}')
        return self

    @model_validator(mode="after")
    def validate_tokenizer_path(self):
        tokenizer_path_proposal = Path(self.tokenizer_path)
        model_name, model_name_param = get_model_name_param(self.model_path)
        expected_path = Path(MODEL_REGISTRY[model_name][model_name_param]["tokenizer_path"])
        if tokenizer_path_proposal.name != expected_path.name:
            raise ValueError(
                f"Tokenizer path is not valid, got {tokenizer_path_proposal}. "
                f"Did you mean {expected_path}?"
            )
        file_root_path = Path(__file__).resolve().parent
        full_tokenizer_path = (file_root_path / "models" / tokenizer_path_proposal).resolve()
        if not full_tokenizer_path.exists():
            raise ValueError(
                f"Path for tokenizer is incorrect or file does not exist: {full_tokenizer_path}"
            )
        self.tokenizer_path = str(full_tokenizer_path)
        return self

settings = Settings()

print(settings.model_dump())