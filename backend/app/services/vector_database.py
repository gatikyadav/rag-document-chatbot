import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np
from datetime import datetime

from ..config import settings
from ..models.schemas import DocumentChunk, SourceCitation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = None, collection_name: str = None, embedding_model: str = None):
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.collection_name
        self.embedding_model_name = embedding_model or settings.embedding_model
        
        logger.info(f"Initializing VectorDatabase with persist_directory={self.persist_directory}")
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
        
        # Initialize embedding model
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' ready")
        except Exception as e:
            logger.error(f"Failed to initialize collection: {str(e)}")
            raise
    
    def add_documents(self, document_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Add document chunks to the vector database."""
        if not document_chunks:
            logger.warning("No document chunks to add")
            return {"added": 0, "failed": 0, "errors": []}
        
        logger.info(f"Adding {len(document_chunks)} document chunks to vector database")
        
        # Prepare data for ChromaDB
        texts = []
        metadatas = []
        ids = []
        added_count = 0
        failed_count = 0
        errors = []
        
        for chunk in document_chunks:
            try:
                # Create unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # Extract text
                texts.append(chunk.text)
                
                # Prepare metadata (ChromaDB has limitations on metadata types)
                metadata = {
                    "source_file": chunk.metadata.source_file,
                    "filename": chunk.metadata.filename,
                    "file_type": chunk.metadata.file_type,
                    "chunk_id": str(chunk.metadata.chunk_id),
                    "public_url": chunk.metadata.public_url,
                    "file_size": chunk.metadata.file_size or "",
                }
                
                # Add optional metadata fields if they exist
                if chunk.metadata.page_number:
                    metadata["page_number"] = str(chunk.metadata.page_number)
                if chunk.metadata.slide_number:
                    metadata["slide_number"] = str(chunk.metadata.slide_number)
                if chunk.metadata.sheet_name:
                    metadata["sheet_name"] = chunk.metadata.sheet_name
                if chunk.metadata.section:
                    metadata["section"] = chunk.metadata.section
                
                metadatas.append(metadata)
                added_count += 1
                
            except Exception as e:
                failed_count += 1
                error_msg = f"Failed to prepare chunk {chunk.metadata.filename}:{chunk.metadata.chunk_id}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        if not texts:
            logger.error("No valid texts to add to database")
            return {"added": 0, "failed": failed_count, "errors": errors}
        
        try:
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode(
                texts, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Add to collection
            logger.info("Adding embeddings to ChromaDB collection...")
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(texts)} document chunks to vector database")
            return {
                "added": len(texts),
                "failed": failed_count,
                "errors": errors,
                "total_documents": self.get_collection_count()
            }
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector database: {str(e)}")
            raise
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        try:
            logger.info(f"Searching for: '{query}' (top {n_results} results)")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "similarity_score": 1.0 - results["distances"][0][i]  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def search_with_citations(self, query: str, n_results: int = 5) -> List[SourceCitation]:
        """Search and return results as SourceCitation objects."""
        search_results = self.search(query, n_results)
        
        citations = []
        for result in search_results:
            metadata = result["metadata"]
            
            citation = SourceCitation(
                filename=metadata.get("filename", "Unknown"),
                file_type=metadata.get("file_type", "unknown"),
                url=metadata.get("public_url", ""),
                relevance_score=result["similarity_score"],
                snippet=result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                page=int(metadata["page_number"]) if metadata.get("page_number") else None,
                slide_number=int(metadata["slide_number"]) if metadata.get("slide_number") else None,
                sheet=metadata.get("sheet_name"),
                section=metadata.get("section")
            )
            citations.append(citation)
        
        return citations
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory,
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            raise
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to get collection count: {str(e)}")
            return 0
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            logger.info(f"Clearing collection '{self.collection_name}'")
            
            # Delete and recreate collection
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Collection cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {str(e)}")
            return False
    
    def delete_documents_by_source(self, source_file: str) -> int:
        """Delete documents by source file."""
        try:
            logger.info(f"Deleting documents from source: {source_file}")
            
            # Get all documents with matching source
            results = self.collection.get(
                where={"source_file": source_file},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info(f"Deleted {deleted_count} documents from source: {source_file}")
                return deleted_count
            else:
                logger.info(f"No documents found for source: {source_file}")
                return 0
                
        except Exception as e:
            logger.error(f"Failed to delete documents by source: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the vector database."""
        try:
            # Test basic operations
            count = self.collection.count()
            
            # Test embedding generation
            test_text = "Health check test"
            test_embedding = self.embedding_model.encode([test_text])
            
            return {
                "status": "healthy",
                "collection_count": count,
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": len(test_embedding[0]),
                "client_version": chromadb.__version__
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }