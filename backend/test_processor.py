# test_processor.py
import sys
import os

# Add the current directory to Python path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_processor import DocumentProcessor

def test_document_processor():
    print("🧪 Testing Document Processor...")
    
    processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    # Create test directory if it doesn't exist
    test_dir = "./documents/text"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test file path
    test_file = os.path.join(test_dir, "test_document.txt")
    
    # Create test document if it doesn't exist
    if not os.path.exists(test_file):
        print("📝 Creating test document...")
        with open(test_file, 'w') as f:
            f.write("""This is a test document for our RAG chatbot system.

## Introduction
This document contains sample text to test our document processing pipeline.

### Key Features
- Document parsing and text extraction
- Intelligent chunk creation with overlap
- Comprehensive metadata generation
- Support for multiple file formats

### Technical Details
The system should be able to process this text and create meaningful chunks for vector storage.
Each chunk will include metadata about the source file, chunk position, and other relevant information.

### Testing
This test document helps verify that our document processor correctly:
1. Extracts text from files
2. Splits text into manageable chunks
3. Creates proper metadata for each chunk
4. Handles various document formats

The RAG system will use these processed chunks to provide accurate answers with source citations.""")
    
    try:
        print(f"📄 Processing file: {test_file}")
        chunks = processor.process_file(test_file)
        print(f"✅ Successfully processed document: {len(chunks)} chunks created")
        
        # Display chunk information
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Text length: {len(chunk.text)} characters")
            print(f"Text preview: {chunk.text[:100]}...")
            print(f"Filename: {chunk.metadata.filename}")
            print(f"File Type: {chunk.metadata.file_type}")
            print(f"Chunk ID: {chunk.metadata.chunk_id}")
            print(f"Public URL: {chunk.metadata.public_url}")
            print(f"File Size: {chunk.metadata.file_size}")
            
        print(f"\n📊 Summary:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Average chunk size: {sum(len(c.text) for c in chunks) // len(chunks)} characters")
        print(f"   Supported extensions: {processor.get_supported_extensions()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_document_processor()
    if success:
        print("\n🎉 Document processor test completed successfully!")
    else:
        print("\n💥 Document processor test failed!")