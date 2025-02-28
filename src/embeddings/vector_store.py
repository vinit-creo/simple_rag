from langchain.vectorstores import Chroma
from typing import List, Dict, Any
from src.config.config import get_config
from src.config.logging import get_logger

logger = get_logger(__name__)

class VectorStore:
    """
    Manage vector storage and retrieval operations.
    """
    
    def __init__(self, embedding_model):
        """
        Initialize the vector store.
        
        Args:
            embedding_model: Model to create embeddings
        """
        self.embedding_model = embedding_model
        self.config = get_config()
        self.persist_directory = self.config.get("vector_store_dir", "./vector_store")
        
    def create_from_documents(self, documents: List[Dict[str, Any]]):
        """
        Create a vector store from document chunks.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
            
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Creating vector store from {len(documents)} documents")
        
        # Convert our document format to langchain document format
        from langchain.schema import Document
        langchain_docs = [
            Document(page_content=doc["content"], metadata=doc["metadata"])
            for doc in documents
        ]
        
        # Create and persist the vector store
        vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        vector_store.persist()
        logger.info(f"Vector store created and persisted to {self.persist_directory}")
        
        return vector_store
    
    def load(self):
        """
        Load an existing vector store.
        
        Returns:
            Chroma vector store instance
        """
        logger.info(f"Loading vector store from {self.persist_directory}")
        
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        
        return vector_store