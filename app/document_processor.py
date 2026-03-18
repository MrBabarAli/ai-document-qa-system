# app/document_processor.py
"""
Document Processor Module
Handles text cleaning and chunking for better QA performance
"""

import re
from typing import List, Dict, Any
import hashlib

class DocumentProcessor:
    """Process and prepare document text for QA system"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the document processor
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"✅ DocumentProcessor initialized (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove multiple tabs
        text = re.sub(r'\t+', '\t', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])- ([a-z])', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\.([A-Z])', r'. \1', text)  # Add space after periods
        
        return text.strip()
    
    def split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Cleaned text
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # If text is shorter than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        start = 0
        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to find a good breaking point (period, newline, space)
            break_points = [
                text.rfind('. ', start, end),
                text.rfind('\n', start, end),
                text.rfind(' ', start, end)
            ]
            
            # Find the best break point (highest positive value)
            best_break = max([bp for bp in break_points if bp != -1] + [end])
            
            if best_break > start:
                chunks.append(text[start:best_break + 1])
                start = best_break + 1 - self.chunk_overlap
            else:
                # No good break point, force split at chunk_size
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
            
            # Ensure we're making progress
            if start < 0:
                start = 0
        
        return chunks
    
    def create_chunks_with_metadata(self, text: str, source: str, page_num: int = None) -> List[Dict[str, Any]]:
        """
        Create chunks with metadata for vector database
        
        Args:
            text: Document text
            source: Source file name
            page_num: Page number (optional)
            
        Returns:
            List of dictionaries with chunk text and metadata
        """
        # Clean the text first
        cleaned_text = self.clean_text(text)
        
        # Split into chunks
        chunks = self.split_into_chunks(cleaned_text)
        
        # Create chunks with metadata
        chunked_docs = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():  # Only add non-empty chunks
                chunk_data = {
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "char_length": len(chunk_text)
                    }
                }
                
                # Add page number if available
                if page_num:
                    chunk_data["metadata"]["page"] = page_num
                
                # Create a unique ID for the chunk
                unique_string = f"{source}_{page_num}_{i}_{chunk_text[:50]}"
                chunk_data["id"] = hashlib.md5(unique_string.encode()).hexdigest()
                
                chunked_docs.append(chunk_data)
        
        return chunked_docs
    
    def process_document(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Main method to process a complete document
        
        Args:
            text: Extracted text from document
            filename: Name of the source file
            
        Returns:
            List of processed chunks with metadata
        """
        print(f"📄 Processing document: {filename}")
        print(f"   Original text length: {len(text)} characters")
        
        # Process the document
        chunks = self.create_chunks_with_metadata(text, filename)
        
        print(f"   Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(c['text']) for c in chunks) // len(chunks)} characters")
        
        return chunks


# Test the processor
if __name__ == "__main__":
    print("🔧 Testing DocumentProcessor...")
    
    # Create processor
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=20)
    
    # Test text
    test_text = """
    This is a sample document for testing the document processor.
    It has multiple sentences and paragraphs.
    
    This is the second paragraph. It contains important information
    about how the document processor works. The processor should split
    this text into chunks while maintaining context.
    
    Finally, this is the third paragraph with more content to ensure
    that our chunking strategy works correctly with longer documents.
    """
    
    # Process the test document
    chunks = processor.process_document(test_text, "test_document.txt")
    
    # Display results
    print("\n📊 Chunking Results:")
    print("-" * 40)
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1}:")
        print(f"ID: {chunk['id']}")
        print(f"Text: {chunk['text'][:100]}...")
        print(f"Metadata: {chunk['metadata']}")
    
    print("\n✅ DocumentProcessor is working!")