from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os
from datetime import datetime

from .config import settings, validate_api_keys, get_active_llm_provider
from .models.schemas import HealthResponse, ErrorResponse
from .api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting RAG Chatbot Backend...")
    
    # Validate configuration
    if not validate_api_keys():
        logger.warning("No valid API keys found. LLM features may not work.")
    
    # Create necessary directories
    os.makedirs(settings.documents_path, exist_ok=True)
    os.makedirs(settings.static_documents_path, exist_ok=True)
    os.makedirs(settings.data_path, exist_ok=True)
    os.makedirs(settings.chroma_persist_directory, exist_ok=True)
    
    logger.info(f"Using LLM provider: {get_active_llm_provider()}")
    logger.info(f"Using embedding model: {settings.embedding_model}")
    
    logger.info("Backend startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot Backend...")

# Create FastAPI application
app = FastAPI(
    title="RAG Document Chatbot API",
    description="A Retrieval-Augmented Generation chatbot for document querying with source citations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files for document serving
if os.path.exists("./static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["RAG API"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "RAG Document Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "detail": str(exc) if settings.debug else "An unexpected error occurred",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )