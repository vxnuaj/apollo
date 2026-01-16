from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.config import settings
from server.routes import router
from server.dependencies import set_tokenizer
from database.connection import init_db, close_db

BLUE = "\033[34m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    
    print(f"{BLUE}INITIALIZING:{RESET} Initializing database...")
    await init_db()
    print(f"{BLUE}INITIALIZING:{RESET} Database initialized")
    print(f"{BLUE}INITIALIZING:{RESET} Loading model from {settings.model_path}...")
    try: # TODO - make this model-agnostic
        from models.tokenizer import GemmaTokenizer
        print(f"{BLUE}INITIALIZING:{RESET} Loading tokenizer from {settings.tokenizer_path}...")
        tokenizer = GemmaTokenizer(settings.tokenizer_path)
        set_tokenizer(tokenizer)
        print(f"{BLUE}INITIALIZING:{RESET} Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    except Exception as e:
        print(f"{RED}INITIALIZING:{RESET} Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"{GREEN}INFO:{RESET}     Server ready on http://{settings.host}:{settings.port}")
    print(f"{GREEN}INFO:{RESET}     API docs: http://{settings.host}:{settings.port}/docs")
    yield
    print("Shutting down Apollo Inference Server...")
    await close_db()
    print("Shutdown complete")

app = FastAPI(
    title="Apollo API",
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
