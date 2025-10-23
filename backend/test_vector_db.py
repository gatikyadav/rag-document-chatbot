# test_vector_db.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_processor import DocumentProcessor
from app.services.vector_database import VectorDatabase

def test_vector_database():
    print("🧪 Testing Vector Database...")
    
    # Initialize components
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    vector_db = VectorDatabase()
    
    # Create test document if it doesn't exist
    test_dir = "./documents/text"
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "vector_test.txt")
    
    if not os.path.exists(test_file):
        print("📝 Creating test document for vector database...")
        with open(test_file, 'w') as f:
            f.write("""RAG Document Chatbot System

Introduction
This is a comprehensive document about Retrieval-Augmented Generation (RAG) systems.
RAG combines the power of information retrieval with language generation.

Technical Architecture
Our system uses ChromaDB for vector storage and OpenAI for language generation.
The document processor handles multiple file formats including PDF, DOCX, and PPTX.

Vector Embeddings
We use sentence transformers to convert text into high-dimensional vectors.
These embeddings capture semantic meaning and enable similarity search.

Query Processing
When users ask questions, we:
1. Convert the question to an embedding
2. Search for similar document chunks
3. Generate answers using retrieved context
4. Provide source citations for transparency

Benefits
- Accurate answers based on your documents
- Source attribution for verification
- Support for multiple file formats
- Real-time processing and search""")
    
    try:
        # Test 1: Process documents
        print("\n📄 Step 1: Processing document...")
        chunks = processor.process_file(test_file)
        print(f"✅ Created {len(chunks)} document chunks")
        
        # Test 2: Add to vector database
        print("\n📊 Step 2: Adding chunks to vector database...")
        result = vector_db.add_documents(chunks)
        print(f"✅ Added {result['added']} chunks to vector database")
        if result['failed'] > 0:
            print(f"⚠️  {result['failed']} chunks failed")
        
        # Test 3: Get collection info
        print("\n📈 Step 3: Getting collection info...")
        info = vector_db.get_collection_info()
        print(f"✅ Collection: {info['collection_name']}")
        print(f"   Documents: {info['document_count']}")
        print(f"   Model: {info['embedding_model']}")
        
        # Test 4: Search functionality
        print("\n🔍 Step 4: Testing search...")
        test_queries = [
            "What is RAG?",
            "How does vector embedding work?",
            "What file formats are supported?",
            "How does query processing work?"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            results = vector_db.search(query, n_results=2)
            
            if results:
                for i, result in enumerate(results):
                    print(f"   Result {i+1}:")
                    print(f"     Similarity: {result['similarity_score']:.3f}")
                    print(f"     Text: {result['text'][:100]}...")
                    print(f"     Source: {result['metadata']['filename']}")
            else:
                print("   No results found")
        
        # Test 5: Health check
        print("\n🏥 Step 5: Health check...")
        health = vector_db.health_check()
        print(f"✅ Status: {health['status']}")
        print(f"   Embedding dimension: {health.get('embedding_dimension', 'Unknown')}")
        
        print("\n🎉 Vector database test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing vector database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_database()
    if success:
        print("\n✅ All vector database tests passed!")
    else:
        print("\n💥 Vector database tests failed!")