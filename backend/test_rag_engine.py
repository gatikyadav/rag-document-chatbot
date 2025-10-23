# test_rag_engine.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_processor import DocumentProcessor
from app.services.vector_database import VectorDatabase
from app.services.rag_engine import RAGEngine

def test_rag_engine():
    print("🧪 Testing Complete RAG Pipeline...")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
    vector_db = VectorDatabase()
    rag_engine = RAGEngine(vector_db)
    
    # Check if we have OpenAI configured
    model_info = rag_engine.get_model_info()
    print(f"   LLM Provider: {model_info['llm_provider']}")
    print(f"   LLM Model: {model_info['llm_model']}")
    
    # Create comprehensive test document
    test_dir = "./documents/text"
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "rag_test.txt")
    
    print("\n📝 Creating comprehensive test document...")
    with open(test_file, 'w') as f:
        f.write("""Company Financial Report Q4 2024

Executive Summary
Our company achieved record revenue of $50 million in Q4 2024, representing 25% growth year-over-year.
The strong performance was driven by increased demand for our AI products and expansion into new markets.

Revenue Breakdown
- AI Products: $30 million (60% of total revenue)
- Cloud Services: $15 million (30% of total revenue)  
- Consulting: $5 million (10% of total revenue)

Geographic Performance
North America generated $35 million (70% of revenue), while Europe contributed $10 million (20%) and Asia-Pacific $5 million (10%).

Key Metrics
- Customer acquisition cost decreased by 15% to $200 per customer
- Customer lifetime value increased to $2,500
- Monthly recurring revenue grew 30% to $4.2 million
- Employee headcount reached 150 people

Technology Initiatives
We launched our new RAG-based chatbot system that helps customers query their documents.
The system uses OpenAI GPT models for natural language generation and ChromaDB for vector storage.
Customer satisfaction with the new system increased by 40%.

Market Outlook
We expect continued growth in 2025, with projected revenue of $75 million.
The AI market is expanding rapidly, and we are well-positioned to capture this growth.
Our main competitors include TechCorp and DataSoft, but we have a competitive advantage in document processing.

Risk Factors
- Potential regulatory changes in AI
- Increased competition in the market
- Supply chain disruptions
- Cybersecurity threats""")
    
    try:
        # Step 1: Process and add document to vector database
        print("\n📊 Processing and indexing document...")
        chunks = processor.process_file(test_file)
        result = vector_db.add_documents(chunks)
        print(f"✅ Added {result['added']} chunks to vector database")
        
        # Step 2: Test various types of questions
        test_questions = [
            "What was the total revenue in Q4 2024?",
            "How much did AI products contribute to revenue?", 
            "What is the customer acquisition cost?",
            "Which technology initiatives were launched?",
            "What are the main risk factors?",
            "How did North America perform compared to other regions?",
            "What is the projected revenue for 2025?",
            "Who are the main competitors?"
        ]
        
        print(f"\n🤔 Testing {len(test_questions)} questions...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i}: {question} ---")
            
            try:
                response = rag_engine.ask_question(question, max_sources=3)
                
                print(f"Answer: {response.answer}")
                print(f"Confidence: {response.confidence:.3f}")
                print(f"Processing time: {response.processing_time:.2f}s")
                print(f"Sources: {len(response.sources)}")
                
                # Show top sources
                for j, source in enumerate(response.sources[:2]):
                    print(f"  Source {j+1}: {source.filename} (relevance: {source.relevance_score:.3f})")
                    if source.snippet:
                        print(f"    Snippet: {source.snippet[:100]}...")
                        
            except Exception as e:
                print(f"❌ Error with question: {str(e)}")
        
        # Step 3: Health check
        print(f"\n🏥 RAG Engine Health Check...")
        health = rag_engine.health_check()
        print(f"Status: {health['status']}")
        print(f"LLM Available: {health['llm_available']}")
        print(f"Vector DB Status: {health['vector_db']['status']}")
        
        print(f"\n🎉 RAG pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_engine()
    if success:
        print(f"\n✅ All RAG pipeline tests passed!")
    else:
        print(f"\n💥 RAG pipeline tests failed!")