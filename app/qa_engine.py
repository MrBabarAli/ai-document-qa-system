# app/qa_engine.py
"""
Question Answering Engine
Combines retrieval and generation to answer questions based on documents
"""

from app.embeddings import EmbeddingsGenerator
from app.vector_store import VectorStore
from app.document_processor import DocumentProcessor
from typing import List, Dict, Any, Optional
import numpy as np

class QAEngine:
    """Main QA Engine that processes questions and returns answers"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the QA Engine with all components
        
        Args:
            persist_directory: Directory for vector database
        """
        print("🚀 Initializing QA Engine...")
        
        # Initialize components
        self.embedder = EmbeddingsGenerator()
        self.vector_store = VectorStore(persist_directory=persist_directory)
        self.processor = DocumentProcessor()
        
        print("✅ QA Engine ready!")
    
    def add_document(self, text: str, filename: str) -> List[str]:
        """
        Add a document to the knowledge base
        
        Args:
            text: Document text
            filename: Source filename
            
        Returns:
            List of chunk IDs
        """
        print(f"\n📄 Adding document: {filename}")
        
        # Process document into chunks
        chunks = self.processor.process_document(text, filename)
        
        # Extract texts and metadata
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Generate embeddings
        print("   Generating embeddings...")
        embeddings = self.embedder.generate_embeddings(texts)
        
        # Store in vector database
        ids = self.vector_store.add_documents(texts, embeddings, metadatas)
        
        print(f"✅ Document added with {len(ids)} chunks")
        return ids
    
    def ask(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get answer based on documents
        
        Args:
            question: User's question
            n_results: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n❓ Question: {question}")
        
        # Generate embedding for question
        question_embedding = self.embedder.generate_embedding(question)
        
        # Search for relevant chunks
        results = self.vector_store.search(question_embedding, n_results=n_results)
        
        # Format results
        relevant_chunks = []
        context_text = ""
        
        if results and len(results['ids'][0]) > 0:
            for i, (text, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                similarity = 1 - distance  # Convert distance to similarity
                relevant_chunks.append({
                    'text': text,
                    'metadata': metadata,
                    'similarity': similarity
                })
                context_text += f"\n[Chunk {i+1} from {metadata['source']}]:\n{text}\n"
            
            # Generate answer (simple extractive approach for now)
            answer = self._generate_answer(question, relevant_chunks)
        else:
            answer = "No relevant documents found to answer this question."
            relevant_chunks = []
        
        response = {
            'question': question,
            'answer': answer,
            'sources': relevant_chunks,
            'num_chunks_used': len(relevant_chunks)
        }
        
        return response
    
    def _generate_answer(self, question: str, chunks: List[Dict]) -> str:
        """
        Generate answer from relevant chunks
        For now, we'll use the most relevant chunk
        Later we can integrate with an LLM
        """
        if not chunks:
            return "I couldn't find any relevant information."
        
        # Get the most relevant chunk
        best_chunk = chunks[0]
        
        # Simple answer construction
        answer = f"Based on the document '{best_chunk['metadata']['source']}', "
        answer += f"I found this information:\n\n{best_chunk['text']}"
        
        if len(chunks) > 1:
            answer += f"\n\n(Found {len(chunks)} relevant sections)"
        
        return answer
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'vector_store': self.vector_store.get_collection_stats(),
            'embedder': self.embedder.get_model_info()
        }
    
    def clear_knowledge_base(self):
        """Clear all documents from the knowledge base"""
        self.vector_store.clear_all()
        print("🧹 Knowledge base cleared")

# Test the QA Engine
if __name__ == "__main__":
    print("🔧 Testing QA Engine...")
    print("=" * 60)
    
    # Create QA Engine
    qa = QAEngine(persist_directory="./test_qa_db")
    
    # Add a test document
    test_doc = """
    Machine Learning is a subset of artificial intelligence that enables systems to learn from data.
    Deep Learning is a further subset of machine learning using neural networks with multiple layers.
    Natural Language Processing (NLP) helps computers understand and generate human language.
    Computer Vision enables machines to interpret and understand visual information from the world.
    Reinforcement Learning is about training agents to make decisions through trial and error.
    """
    
    qa.add_document(test_doc, "ai_concepts.txt")
    
    # Test questions
    questions = [
        "What is machine learning?",
        "What is deep learning?",
        "What does NLP do?",
        "What is reinforcement learning?"
    ]
    
    for question in questions:
        response = qa.ask(question, n_results=2)
        print(f"\n📝 Answer: {response['answer']}")
        print("-" * 40)
    
    # Clean up
    qa.clear_knowledge_base()
    
    print("\n" + "=" * 60)
    print("✅ QA Engine test complete!")
    print("=" * 60)