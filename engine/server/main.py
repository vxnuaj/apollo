"""
Apollo Inference Server - FastAPI application

Run with:
    uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import settings
from server.routes import router
from server.dependencies import set_model, set_tokenizer
from database.connection import init_db, close_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    print("Starting Apollo Inference Server...")
    
    print("Initializing database...")
    await init_db()
    print("Database initialized")
    
    print(f"Loading model from {settings.model_path}...")
    try:
        from models.gemma.helpers.basic_inference import (
            load_gemma_checkpoint,
            restructure_checkpoint,
            create_gemma_config
        )
        from models.gemma._gemma import Gemma
        from models.gemma.helpers.test_tokenizer import GemmaTokenizer
        
        print(f"Loading tokenizer from {settings.tokenizer_path}...")
        tokenizer = GemmaTokenizer(settings.tokenizer_path)
        set_tokenizer(tokenizer)
        print(f"Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
        
        print("Loading checkpoint...")
        flat_checkpoint = load_gemma_checkpoint(settings.model_path)
        checkpoint = restructure_checkpoint(flat_checkpoint)
        print(f"Checkpoint loaded ({len(flat_checkpoint)} parameter groups)")
        
        print("Initializing model...")
        config = create_gemma_config()
        model = Gemma(config=config)
        
        set_model(checkpoint)
        
        import server.dependencies as deps
        deps._model_structure = model
        
        print("Model initialized")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"Server ready on http://{settings.host}:{settings.port}")
    print(f"API docs: http://{settings.host}:{settings.port}/docs")
    
    yield
    
    print("Shutting down Apollo Inference Server...")
    await close_db()
    print("Shutdown complete")


app = FastAPI(
    title="Apollo Inference API",
    description="REST API for Apollo LLM inference with Gemma-3-270M",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Apollo Inference API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
