# process_all_documents.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.document_processor import DocumentProcessor
from app.services.vector_database import VectorDatabase

def process_all_documents():
    print("Processing all documents in ./documents/ directory...")
    
    processor = DocumentProcessor()
    vector_db = VectorDatabase()
    
    documents_base = "./documents"
    total_processed = 0
    total_chunks = 0
    
    # Get current collection count
    initial_count = vector_db.get_collection_count()
    print(f"Initial collection size: {initial_count} documents")
    
    # Process each subdirectory
    for subdir in ["text", "pdf", "docx", "pptx", "xlsx", "html"]:
        subdir_path = os.path.join(documents_base, subdir)
        
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Created directory: {subdir_path}")
            continue
            
        files = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]
        
        if not files:
            print(f"No files in {subdir}/ directory")
            continue
            
        print(f"\nProcessing {subdir}/ directory ({len(files)} files)...")
        
        for filename in files:
            file_path = os.path.join(subdir_path, filename)
            
            try:
                print(f"  Processing: {filename}")
                chunks = processor.process_file(file_path)
                
                if chunks:
                    result = vector_db.add_documents(chunks)
                    chunks_added = result.get('added', 0)
                    total_chunks += chunks_added
                    total_processed += 1
                    
                    print(f"    ✅ {chunks_added} chunks added")
                    
                    if result.get('failed', 0) > 0:
                        print(f"    ⚠️ {result['failed']} chunks failed")
                else:
                    print(f"    ⚠️ No content extracted")
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
    
    # Final summary
    final_count = vector_db.get_collection_count()
    print(f"\n📊 Processing Summary:")
    print(f"   Files processed: {total_processed}")
    print(f"   Total chunks added: {total_chunks}")
    print(f"   Collection size: {initial_count} → {final_count}")
    print(f"   New documents: {final_count - initial_count}")
    
    # Show collection info
    info = vector_db.get_collection_info()
    print(f"\n📈 Collection Info:")
    print(f"   Collection: {info['collection_name']}")
    print(f"   Total documents: {info['document_count']}")
    print(f"   Embedding model: {info['embedding_model']}")

if __name__ == "__main__":
    process_all_documents()