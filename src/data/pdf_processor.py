import os
import fitz  
from typing import List, Dict, Any
from src.config.logging import get_logger

logger = get_logger(__name__)

class PDFProcessor:
    """
    Extract and process text from PDF files.
    """
    
    def __init__(self, pdf_dir: str):
        """
        Initialize the PDF processor.
        
        Args:
            pdf_dir: Directory containing PDF files
        """
        self.pdf_dir = pdf_dir
        
    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """
        Process all PDF files in the directory.
        
        Returns:
            List of documents with text content and metadata
        """
        documents = []
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.lower().endswith('.pdf')]
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for filename in pdf_files:
            try:
                pdf_path = os.path.join(self.pdf_dir, filename)
                content, metadata = self._extract_from_pdf(pdf_path)
                
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": filename,
                        **metadata
                    }
                })
                
                logger.info(f"Processed PDF: {filename}")
                
            except Exception as e:
                logger.error(f"Error processing PDF {filename}: {str(e)}")
                
        return documents
    
    def _extract_from_pdf(self, pdf_path: str) -> tuple:
        """
        Extract text and metadata from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (content, metadata)
        """
        
        print("::: show pdf path " , pdf_path)
        doc = fitz.open(pdf_path)
        

        
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            try:
                text = page.get_text()
                print("::: full text ", full_text)
            except AttributeError:
                try:
                    text = page.getText()
                except AttributeError:
                    # Last resort for very old versions
                    text = page.extractText()
                    
            full_text += f"{text}\n"

        
        return full_text 