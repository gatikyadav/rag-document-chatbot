import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings and configuration."""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # Vector Database Configuration
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    
    # Document Processing Configuration
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "2000"))
    
    # Server Configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # CORS Configuration
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    # File Configuration
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
    allowed_file_types: List[str] = [".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".html"]
    
    # Storage Paths
    documents_path: str = "./documents"
    static_documents_path: str = "./static/documents"
    data_path: str = "./data"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()

# Validation functions
def validate_api_keys() -> bool:
    """Validate that at least one API key is configured."""
    if not settings.openai_api_key and not settings.anthropic_api_key:
        return False
    
    # Check if keys are not placeholder values
    if (settings.openai_api_key == "your_openai_api_key_here" and 
        settings.anthropic_api_key == "your_anthropic_api_key_here"):
        return False
    
    return True

def get_active_llm_provider() -> str:
    """Determine which LLM provider to use based on available keys."""
    if "gpt" in settings.llm_model.lower() and settings.openai_api_key:
        if settings.openai_api_key != "your_openai_api_key_here":
            return "openai"
    
    if "claude" in settings.llm_model.lower() and settings.anthropic_api_key:
        if settings.anthropic_api_key != "your_anthropic_api_key_here":
            return "anthropic"
    
    return "none"