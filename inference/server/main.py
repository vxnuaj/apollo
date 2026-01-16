# inference server

from server.config import settings
from contextlib import asynccontextmanager
import sys
import os
from server.routes import router # TODO - make it lmao
from server.dependencies import set_model
from typing import AsyncContextManager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"{BLUE}INITIALIZING:{RESET} Initializing inference server")
  
    try:
#        from models.gemma._gemma import Gemma
#        from models.gemma._config import GemmaConfig, EmbeddingConfig, TransformerBlockConfig,

#        embedding_config = # TODO
#        transformer_block_config = # TODO
#        num_layers = # TODO
 
#        model_config = GemmaConfig(
#            embedding_config = embedding_config,
#            transformer_block_config = transformer_block_config,
#            num_layers = num_layers
#        )
        

        model = load_model()
   
    except Exception as e:
        print()
        
    
    yield
    
app = FastAPI(
    title = "Inference Server",
    description = "Inference API Endpoints",
    version = "0.1.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = settings.allowed_origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"]
)

app.include_router(router)

app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Apollo Inference Server",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/v1/health",
    }
    
