import PyPDF2
import io
from typing import Optional

class TextExtractor:
    """Extract text from various document formats"""
    
    @staticmethod
    def extract_from_pdf(file_content: bytes) -> str:
        """
        Extract text from PDF file
        Args:
            file_content: PDF file as bytes
        Returns:
            Extracted text as string
        """
        try:
            # Create PDF reader object
            pdf_file = io.BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            
            return text.strip()
        
        except Exception as e:
            return f"Error extracting PDF: {str(e)}"
    
    @staticmethod
    def extract_from_txt(file_content: bytes) -> str:
        """Extract text from TXT file"""
        try:
            return file_content.decode('utf-8')
        except Exception as e:
            return f"Error extracting TXT: {str(e)}"
    
    def extract_text(self, file_content: bytes, file_type: str) -> str:
        """
        Main method to extract text based on file type
        """
        if file_type == "pdf":
            return self.extract_from_pdf(file_content)
        elif file_type == "txt":
            return self.extract_from_txt(file_content)
        else:
            return "Unsupported file type"

# Test the extractor
if __name__ == "__main__":
    # Simple test
    extractor = TextExtractor()
    print("✅ Text Extractor created successfully!")