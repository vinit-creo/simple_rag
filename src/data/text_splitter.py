from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

class DocumentSplitter:
    """
    Split documents into smaller chunks for more effective retrieval.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document splitter.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split a list of documents into chunks.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            List of document chunks with preserved metadata
        """
        chunks = []
        
        for doc in documents:
            content = doc["content"]
            metadata = doc["metadata"]
            
            # Split the text into chunks
            text_chunks = self.text_splitter.split_text(content)
            
            # Create document objects for each chunk with metadata
            for i, chunk in enumerate(text_chunks):
                # Add chunk index to metadata
                chunk_metadata = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_count": len(text_chunks)
                }
                
                chunks.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
                
        return chunks