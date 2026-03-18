# app/embeddings.py
"""
Embeddings Module
Handles conversion of text to vector embeddings using sentence transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch

class EmbeddingsGenerator:
    """Generate embeddings for text chunks"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embeddings generator
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       'all-MiniLM-L6-v2' is fast and good for QA
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        print(f"🔄 Initializing EmbeddingsGenerator with model: {model_name}")
        
    def _load_model(self):
        """Lazy load the model (only loads when needed)"""
        if self.model is None:
            print(f"   Loading model '{self.model_name}'...")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"   ✅ Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts (batch processing)
        
        Args:
            texts: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        
        print(f"   Generating embeddings for {len(texts)} texts...")
        
        # Batch encode for efficiency
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"   ✅ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def generate_embedding_with_metadata(self, text: str, metadata: dict) -> dict:
        """
        Generate embedding and combine with metadata
        
        Args:
            text: Text chunk
            metadata: Associated metadata
            
        Returns:
            Dictionary with text, embedding, and metadata
        """
        embedding = self.generate_embedding(text)
        
        return {
            "text": text,
            "embedding": embedding,
            "metadata": metadata
        }
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        self._load_model()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_seq_length": self.model.max_seq_length,
            "device": str(self.model.device)
        }

# Test the embeddings generator
if __name__ == "__main__":
    print("🔧 Testing EmbeddingsGenerator...")
    
    # Create embeddings generator
    embedder = EmbeddingsGenerator()
    
    # Test texts
    test_texts = [
        "This is the first test sentence about AI.",
        "Machine learning is fascinating.",
        "Document question answering systems use embeddings."
    ]
    
    # Generate single embedding
    print("\n📊 Testing single embedding:")
    single_embedding = embedder.generate_embedding(test_texts[0])
    print(f"   Single embedding shape: {single_embedding.shape}")
    print(f"   First 5 values: {single_embedding[:5]}")
    
    # Generate batch embeddings
    print("\n📊 Testing batch embeddings:")
    batch_embeddings = embedder.generate_embeddings(test_texts)
    print(f"   Batch embeddings shape: {batch_embeddings.shape}")
    
    # Get model info
    print("\n📊 Model Information:")
    model_info = embedder.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ EmbeddingsGenerator is working!")