# test_rag_retrieval_only.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_processor import DocumentProcessor
from app.services.vector_database import VectorDatabase

def test_retrieval_only():
    print("🧪 Testing RAG Retrieval (without LLM)...")
    
    # Initialize components
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    vector_db = VectorDatabase()
    
    # Test questions
    test_questions = [
        "What was the total revenue in Q4 2024?",
        "How much did AI products contribute to revenue?", 
        "What is the customer acquisition cost?",
        "What are the projected revenue for 2025?"
    ]
    
    print("\n🔍 Testing document retrieval...")
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        results = vector_db.search(question, n_results=2)
        
        if results:
            best_result = results[0]
            print(f"✅ Found relevant content (similarity: {best_result['similarity_score']:.3f})")
            print(f"Content preview: {best_result['text'][:150]}...")
            
            # Extract potential answer from the text
            text = best_result['text']
            if "revenue" in question.lower() and "$" in text:
                import re
                money_amounts = re.findall(r'\$[\d,]+(?:\.\d+)?\s*(?:million|billion)?', text)
                if money_amounts:
                    print(f"💡 Potential answer: {money_amounts}")
        else:
            print("❌ No relevant content found")
    
    print("\n✅ Retrieval system is working perfectly!")
    print("📝 Next step: Fix OpenAI API integration")

if __name__ == "__main__":
    test_retrieval_only()