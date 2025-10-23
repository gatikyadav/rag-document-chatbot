from fastapi import APIRouter, HTTPException, Depends, Form
from typing import List, Optional
import logging
from datetime import datetime

from ..models.schemas import (
    QuestionRequest, 
    AnswerResponse, 
    CollectionInfo,
    ErrorResponse
)
from ..config import settings
from ..services.document_processor import DocumentProcessor
from ..services.vector_database import VectorDatabase
from ..services.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

# Global service instances (lazy initialization)
document_processor = None
vector_database = None
rag_engine = None

def get_document_processor():
    """Dependency to get document processor instance."""
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor

def get_vector_database():
    """Dependency to get vector database instance."""
    global vector_database
    if vector_database is None:
        vector_database = VectorDatabase()
    return vector_database

def get_rag_engine():
    """Dependency to get RAG engine instance."""
    global rag_engine
    if rag_engine is None:
        vector_db = get_vector_database()
        rag_engine = RAGEngine(vector_db)
    return rag_engine

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
    question: str = Form(..., description="The question to ask about the documents"),
    max_sources: int = Form(default=5, description="Maximum number of sources to return"),
    rag_engine: RAGEngine = Depends(get_rag_engine)
):
    """
    Ask a question about the documents.
    
    This endpoint will:
    1. Process the question
    2. Retrieve relevant document chunks
    3. Generate an answer using LLM
    4. Return answer with source citations
    """
    try:
        logger.info(f"API: Processing question: {question}")
        
        if not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if max_sources < 1 or max_sources > 10:
            raise HTTPException(status_code=400, detail="max_sources must be between 1 and 10")
        
        # Process question through RAG pipeline
        response = rag_engine.ask_question(
            question=question,
            max_sources=max_sources,
            include_snippets=True
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collection-info", response_model=CollectionInfo)
async def get_collection_info(vector_db: VectorDatabase = Depends(get_vector_database)):
    """Get information about the document collection."""
    try:
        info = vector_db.get_collection_info()
        return CollectionInfo(
            collection_name=info["collection_name"],
            document_count=info["document_count"],
            embedding_model=info["embedding_model"],
            last_updated=datetime.fromisoformat(info["last_updated"]) if info.get("last_updated") else None,
            storage_path=info["persist_directory"]
        )
    except Exception as e:
        logger.error(f"Error getting collection info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/collection")
async def clear_collection(vector_db: VectorDatabase = Depends(get_vector_database)):
    """Clear all documents from the collection."""
    try:
        success = vector_db.clear_collection()
        if success:
            return {"message": "Collection cleared successfully", "timestamp": datetime.now()}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear collection")
    except Exception as e:
        logger.error(f"Error clearing collection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(rag_engine: RAGEngine = Depends(get_rag_engine)):
    """Comprehensive health check for the RAG system."""
    try:
        health_status = rag_engine.health_check()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))