# app/vector_store.py
"""
Vector Store Module
Manages storing and searching embeddings using ChromaDB (new API)
"""

import chromadb
from chromadb.errors import NotFoundError
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import os
from datetime import datetime

class VectorStore:
    """Manage vector database operations using ChromaDB new API"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store with new ChromaDB API
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client with new API
        print(f"🔄 Initializing VectorStore with persistence: {persist_directory}")
        
        # New ChromaDB client initialization
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=chromadb.Settings(
                anonymized_telemetry=False
            )
        )
        
        # Create or get collection
        self.collection_name = "documents"
        self.collection = self._get_or_create_collection()
        
        print(f"✅ VectorStore initialized with collection: '{self.collection_name}'")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(self.collection_name)
            print(f"   Found existing collection with {collection.count()} documents")
            return collection
        except NotFoundError:
            # Create new collection if it doesn't exist
            print(f"   Creating new collection: '{self.collection_name}'")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        except Exception as e:
            print(f"   Unexpected error: {e}")
            # Fallback - try to create new collection
            print(f"   Attempting to create new collection as fallback...")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, 
                      texts: List[str], 
                      embeddings: List[np.ndarray], 
                      metadatas: List[Dict[str, Any]],
                      ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            texts: List of text chunks
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Convert embeddings to lists (ChromaDB requirement)
        embeddings_list = [embedding.tolist() if isinstance(embedding, np.ndarray) else embedding 
                          for embedding in embeddings]
        
        # Add timestamp to metadata
        for metadata in metadatas:
            metadata['added_at'] = datetime.now().isoformat()
        
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Added {len(texts)} documents to vector store")
        return ids
    
    def search(self, 
               query_embedding: np.ndarray, 
               n_results: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Embedding of the query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Dictionary with results
        """
        # Convert query embedding to list
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=n_results,
            where=filter_metadata
        )
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """Get a specific document by ID"""
        try:
            result = self.collection.get(ids=[doc_id])
            if result and len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'text': result['documents'][0] if result['documents'] else None,
                    'metadata': result['metadatas'][0] if result['metadatas'] else None
                }
        except:
            pass
        return None
    
    def get_all_documents(self) -> List[Dict]:
        """Get all documents in the store"""
        results = self.collection.get()
        
        documents = []
        for i, doc_id in enumerate(results['ids']):
            documents.append({
                'id': doc_id,
                'text': results['documents'][i] if results['documents'] else None,
                'metadata': results['metadatas'][i] if results['metadatas'] else None
            })
        
        return documents
    
    def delete_document(self, doc_id: str):
        """Delete a document by ID"""
        self.collection.delete(ids=[doc_id])
        print(f"✅ Deleted document: {doc_id}")
    
    def clear_all(self):
        """Clear all documents from the store"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Cleared all documents from vector store")
        except Exception as e:
            print(f"❌ Error clearing store: {e}")
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory
        }

# Test the vector store
if __name__ == "__main__":
    print("🔧 Testing VectorStore with new API...")
    print("=" * 60)
    
    try:
        # Create vector store
        print("Creating vector store...")
        vector_store = VectorStore(persist_directory="./test_chroma_db")
        
        # Create test data
        test_texts = [
            "The capital of France is Paris.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language for AI.",
            "Embeddings convert text to numerical vectors.",
            "Vector databases enable semantic search."
        ]
        
        # Create simple test embeddings
        print("\n📊 Creating test embeddings...")
        test_embeddings = [np.random.rand(384) for _ in range(len(test_texts))]
        
        # Create metadata
        test_metadatas = [
            {"source": "test1.txt", "chunk_id": 0},
            {"source": "test1.txt", "chunk_id": 1},
            {"source": "test2.txt", "chunk_id": 0},
            {"source": "test2.txt", "chunk_id": 1},
            {"source": "test3.txt", "chunk_id": 0}
        ]
        
        # Add documents
        print("\n📊 Adding test documents...")
        ids = vector_store.add_documents(test_texts, test_embeddings, test_metadatas)
        print(f"   Document IDs: {ids}")
        
        # Get stats
        print("\n📊 Collection Statistics:")
        stats = vector_store.get_collection_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test search
        print("\n📊 Testing search with random query...")
        query_embedding = np.random.rand(384)
        results = vector_store.search(query_embedding, n_results=3)
        
        print(f"   Found {len(results['ids'][0])} results:")
        for i, (doc_id, text, distance) in enumerate(zip(
            results['ids'][0], 
            results['documents'][0],
            results['distances'][0]
        )):
            print(f"\n   Result {i+1}:")
            print(f"      ID: {doc_id}")
            print(f"      Text: {text[:50]}...")
            print(f"      Distance: {distance:.4f}")
            print(f"      Metadata: {results['metadatas'][0][i]}")
        
        print("\n✅ All tests passed! VectorStore is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)