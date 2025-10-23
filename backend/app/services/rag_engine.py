import logging
import time
from typing import List, Dict, Any, Optional
import openai  # This will be the 0.28 version now
from datetime import datetime

from ..config import settings, get_active_llm_provider
from ..models.schemas import QuestionRequest, AnswerResponse, SourceCitation
from .vector_database import VectorDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """Handles the complete RAG pipeline: retrieval + generation."""
    
    def __init__(self, vector_database: VectorDatabase, llm_model: str = None):
        self.vector_db = vector_database
        self.llm_model = llm_model or settings.llm_model
        self.llm_provider = get_active_llm_provider()
    
        # Initialize OpenAI with legacy format (0.28)
        if self.llm_provider == "openai":
            try:
                openai.api_key = settings.openai_api_key
                logger.info(f"RAG Engine initialized with OpenAI model: {self.llm_model}")
            except Exception as e:
                logger.error(f"Failed to set OpenAI API key: {str(e)}")
                self.llm_provider = "none"
        else:
            logger.warning(f"No valid LLM provider configured. Current provider: {self.llm_provider}")
    
        # RAG configuration
        self.max_context_length = 3000
        self.min_similarity_threshold = 0.1
        
    def ask_question(self, question: str, max_sources: int = 5, include_snippets: bool = True) -> AnswerResponse:
        """
        Process a question through the complete RAG pipeline.
        
        Args:
            question: The user's question
            max_sources: Maximum number of source documents to retrieve
            include_snippets: Whether to include text snippets in citations
            
        Returns:
            AnswerResponse with answer, sources, and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing question: '{question}'")
            
            # Step 1: Validate question
            if not question.strip():
                raise ValueError("Question cannot be empty")
            
            # Step 2: Retrieve relevant documents
            logger.info("Retrieving relevant documents...")
            search_results = self.vector_db.search(question, n_results=max_sources)
            
            if not search_results:
                return self._create_no_results_response(question, start_time)
            
            # Step 3: Filter results by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result.get('similarity_score', 0) >= self.min_similarity_threshold
            ]
            
            if not filtered_results:
                return self._create_low_confidence_response(question, start_time)
            
            logger.info(f"Found {len(filtered_results)} relevant documents")
            
            # Step 4: Prepare context for LLM
            context = self._prepare_context(filtered_results)
            
            # Step 5: Generate answer using LLM
            if self.llm_provider != "openai":
                return self._create_no_llm_response(question, filtered_results, start_time)
            
            answer = self._generate_answer(question, context)
            
            # Step 6: Create source citations
            sources = self._create_source_citations(filtered_results, include_snippets)
            
            # Step 7: Calculate confidence score
            confidence = self._calculate_confidence(filtered_results, answer)
            
            processing_time = time.time() - start_time
            
            response = AnswerResponse(
                question=question,
                answer=answer,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
            logger.info(f"Question processed successfully in {processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            processing_time = time.time() - start_time
            
            return AnswerResponse(
                question=question,
                answer=f"I apologize, but I encountered an error while processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context from search results for the LLM."""
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(search_results):
            # Create source reference
            metadata = result['metadata']
            source_ref = f"Source {i+1} ({metadata.get('filename', 'Unknown')})"
            
            # Add page/section info if available
            location_info = []
            if metadata.get('page_number'):
                location_info.append(f"Page {metadata['page_number']}")
            if metadata.get('slide_number'):
                location_info.append(f"Slide {metadata['slide_number']}")
            if metadata.get('sheet_name'):
                location_info.append(f"Sheet: {metadata['sheet_name']}")
            if metadata.get('section'):
                location_info.append(f"Section: {metadata['section']}")
            
            if location_info:
                source_ref += f" - {', '.join(location_info)}"
            
            # Prepare context entry
            context_entry = f"\n{source_ref}:\n{result['text']}\n"
            
            # Check if adding this would exceed max context length
            if current_length + len(context_entry) > self.max_context_length:
                logger.info(f"Context length limit reached, using first {i} sources")
                break
            
            context_parts.append(context_entry)
            current_length += len(context_entry)
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using OpenAI."""
        prompt = self._create_prompt(question, context)
    
        try:
            response = openai.ChatCompletion.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context from documents. Always base your answers on the given information and cite sources when possible."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=1000,
                top_p=0.9
            )
        
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer with {len(answer)} characters")
            return answer
        
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a well-structured prompt for the LLM."""
        return f"""Based on the following context from documents, please answer the question accurately and comprehensively.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Base your answer strictly on the provided context
- If the information is not available in the context, clearly state this
- When referencing information, mention the source (e.g., "According to Source 1...")
- Be specific and detailed in your response
- If you're uncertain about any information, express that uncertainty
- Provide a clear, well-structured answer

ANSWER:"""
    
    def _create_source_citations(self, search_results: List[Dict[str, Any]], include_snippets: bool = True) -> List[SourceCitation]:
        """Create source citations from search results."""
        citations = []
        
        for result in search_results:
            metadata = result['metadata']
            
            # Create snippet if requested
            snippet = ""
            if include_snippets:
                text = result['text']
                snippet = text[:200] + "..." if len(text) > 200 else text
            
            citation = SourceCitation(
                filename=metadata.get('filename', 'Unknown'),
                file_type=metadata.get('file_type', 'unknown'),
                url=metadata.get('public_url', ''),
                relevance_score=result.get('similarity_score', 0.0),
                snippet=snippet,
                page=int(metadata['page_number']) if metadata.get('page_number') else None,
                slide_number=int(metadata['slide_number']) if metadata.get('slide_number') else None,
                sheet=metadata.get('sheet_name'),
                section=metadata.get('section')
            )
            citations.append(citation)
        
        return citations
    
    def _calculate_confidence(self, search_results: List[Dict[str, Any]], answer: str) -> float:
        """Calculate confidence score for the answer."""
        if not search_results:
            return 0.0
        
        # Base confidence on average similarity scores
        similarities = [result.get('similarity_score', 0.0) for result in search_results]
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost confidence if we have multiple good sources
        source_count_factor = min(len(search_results) / 3.0, 1.0)  # Up to 3 sources give full boost
        
        # Penalize very short answers (might indicate lack of information)
        length_factor = min(len(answer) / 100.0, 1.0)  # 100+ characters = full confidence
        
        # Combine factors
        confidence = avg_similarity * 0.6 + source_count_factor * 0.2 + length_factor * 0.2
        
        # Ensure confidence is between 0.1 and 0.95
        return max(0.1, min(0.95, confidence))
    
    def _create_no_results_response(self, question: str, start_time: float) -> AnswerResponse:
        """Create response when no relevant documents are found."""
        return AnswerResponse(
            question=question,
            answer="I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the relevant documents have been uploaded.",
            sources=[],
            confidence=0.0,
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    def _create_low_confidence_response(self, question: str, start_time: float) -> AnswerResponse:
        """Create response when confidence is too low."""
        return AnswerResponse(
            question=question,
            answer="I found some potentially relevant information, but the similarity scores are too low to provide a confident answer. Please try a more specific question or check if the relevant documents have been uploaded.",
            sources=[],
            confidence=0.0,
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    def _create_no_llm_response(self, question: str, search_results: List[Dict[str, Any]], start_time: float) -> AnswerResponse:
        """Create response when LLM is not available."""
        sources = self._create_source_citations(search_results, include_snippets=True)
        
        return AnswerResponse(
            question=question,
            answer="I found relevant documents for your question, but I cannot generate an answer because the OpenAI client is not properly configured. Please check your OpenAI API key configuration.",
            sources=sources,
            confidence=0.5,
            processing_time=time.time() - start_time,
            timestamp=datetime.now()
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration."""
        return {
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider,
            "embedding_model": self.vector_db.embedding_model_name,
            "max_context_length": self.max_context_length,
            "min_similarity_threshold": self.min_similarity_threshold,
            "openai_available": self.llm_provider == "openai"
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the RAG engine."""
        try:
            # Check vector database
            db_health = self.vector_db.health_check()
        
            # Check LLM availability
            llm_available = self.llm_provider == "openai"  # Changed this line
        
            return {
                "status": "healthy" if db_health["status"] == "healthy" and llm_available else "degraded",
                "vector_db": db_health,
                "llm_available": llm_available,
                "llm_provider": self.llm_provider,
                "model_info": self.get_model_info()
            }
        
        except Exception as e:
            logger.error(f"RAG engine health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }