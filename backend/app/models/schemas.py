from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Metadata for a document chunk."""
    source_file: str
    filename: str
    file_type: str
    chunk_id: int
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    sheet_name: Optional[str] = None
    cell_range: Optional[str] = None
    section: Optional[str] = None
    public_url: str
    file_size: Optional[str] = None
    created_date: Optional[datetime] = None

class DocumentChunk(BaseModel):
    """A processed document chunk."""
    text: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class SourceCitation(BaseModel):
    """Source citation for an answer."""
    filename: str
    file_type: str
    url: str
    relevance_score: float
    snippet: str
    page: Optional[int] = None
    slide_number: Optional[int] = None
    sheet: Optional[str] = None
    cell_range: Optional[str] = None
    section: Optional[str] = None

class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask")
    max_sources: int = Field(default=5, ge=1, le=10, description="Maximum number of sources to return")
    include_snippets: bool = Field(default=True, description="Whether to include text snippets in sources")

class AnswerResponse(BaseModel):
    """Response model for answers."""
    question: str
    answer: str
    sources: List[SourceCitation]
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    timestamp: datetime

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    embedding_model: str
    llm_model: str
    llm_provider: str
    collection_info: Dict[str, Any]

class CollectionInfo(BaseModel):
    """Information about the document collection."""
    collection_name: str
    document_count: int
    embedding_model: str
    last_updated: Optional[datetime] = None
    storage_path: str

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None