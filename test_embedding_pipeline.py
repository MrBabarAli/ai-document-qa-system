# test_embedding_pipeline.py
"""
Test the complete embedding pipeline
"""

import sys
import os
sys.path.append(os.getcwd())

from app.document_processor import DocumentProcessor
from app.embeddings import EmbeddingsGenerator
from app.vector_store import VectorStore

def test_pipeline():
    print("=" * 60)
    print("🔍 TESTING COMPLETE EMBEDDING PIPELINE")
    print("=" * 60)
    
    # Step 1: Create sample document
    print("\n📄 Step 1: Creating sample document")
    sample_text = """
    Artificial Intelligence (AI) is transforming the world.
    Machine learning, a subset of AI, enables computers to learn from data.
    Deep learning uses neural networks with multiple layers.
    Natural Language Processing (NLP) helps computers understand human language.
    Document question answering systems combine NLP with information retrieval.
    These systems can answer questions based on document content.
    They use embeddings to find relevant information quickly.
    Vector databases store these embeddings for efficient similarity search.
    """
    
    # Step 2: Process document
    print("\n🔧 Step 2: Processing document")
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
    chunks = processor.process_document(sample_text, "sample_doc.txt")
    
    # Step 3: Generate embeddings
    print("\n🎯 Step 3: Generating embeddings")
    embedder = EmbeddingsGenerator()
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedder.generate_embeddings(texts)
    
    # Step 4: Store in vector database
    print("\n💾 Step 4: Storing in vector database")
    vector_store = VectorStore(persist_directory="./test_pipeline_db")
    
    metadatas = [chunk['metadata'] for chunk in chunks]
    ids = vector_store.add_documents(texts, embeddings, metadatas)
    
    # Step 5: Test search
    print("\n🔍 Step 5: Testing search")
    test_query = "What is machine learning?"
    print(f"   Query: '{test_query}'")
    
    # Generate query embedding
    query_embedding = embedder.generate_embedding(test_query)
    
    # Search
    results = vector_store.search(query_embedding, n_results=3)
    
    print("\n📊 Search Results:")
    print("-" * 40)
    for i, (text, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\nResult {i+1} (similarity: {1-distance:.3f}):")
        print(f"   Text: {text[:100]}...")
        print(f"   Source: {metadata['source']}, Chunk: {metadata['chunk_id']}")
    
    # Clean up
    print("\n🧹 Cleaning up...")
    vector_store.clear_all()
    
    print("\n" + "=" * 60)
    print("✅ Pipeline test complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()