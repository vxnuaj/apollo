# basic foncig for the inference server - loaded via inferec/.env

import os
import sys
from typing import List
from dotenv import load_dotenv
from pathlib import Path
from pydantic import field_validator, model_validator, Field
from pydantic_settings import SettingsConfigDict, BaseSettings
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.model_registry import MODEL_REGISTRY

load_dotenv()

class Settings(BaseSettings):
    model_name: str # = gemma/gemma-3-270m
    max_context_length: int
    device: int  = 0 # main gpu idx
    precision: str = "fp32"
    host:str =  "0.0.0.0"
    port:str =  "8001"
    max_connection:int = Field(10, ge = 1, le = 100)
    session_timeout:int = Field(600, ge = 600, le = 3600) # in seconds
    allowed_origins: List[str] = []
   
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore" # keep - such that we dont' get disorganiezx with out settings class
    )
    
    @field_validator("model_name", check_fields = "before")
    @classmethod
    def validate_model_name(cls, v:str):
        model_name, model_param_name = v.split("/") 
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"`{model_name}` is not in the MODEL_REGISTRY")
        if model_param_name not in MODEL_REGISTRY[model_name]:
            raise ValueError(f"`{model_param_name}` is not in the MODEL_REGISTRY")
        return v

    @field_validator("precision", check_fields = "before")
    @classmethod
    def validate_precision(cls, v: str):
        p_dtype = ["fp16", "bf16", "fp32"]
        if v not in p_dtype:
            raise ValueError(f"Invalid dtype. Expected {p_dtype.join(", ")}, got {v}")
        return v
    
    @field_validator("allowed_origins", mode="before")
    def validate_allowed_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v
  
    @model_validator(mode = "after")
    def validate_context_length(self):
        model_name, model_param_name = self.model_name.split("/")
        model_context_length = MODEL_REGISTRY[model_name][model_param_name]["context_len"]
        if self.max_context_length > model_context_length:
            raise ValueError(f"model context length exceeds the maximum context length allowed. got {self.max_context_length}, expected a maximum of {model_context_length} for {self.model_name}")
        return self
  
settings = Settings()