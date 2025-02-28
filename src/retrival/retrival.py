from src.config.config import get_config
from src.config.logging import get_logger

logger = get_logger(__name__)

class Retriever:
    """
    Manage document retrieval from vector store.
    """
    
    def __init__(self, vector_store):
        """
        Initialize with vector store.
        
        Args:
            vector_store: Chroma vector store instance
        """
        self.vector_store = vector_store
        self.config = get_config()
        
    def get_retriever(self):
        """
        Create a configured retriever.
        
        Returns:
            Retriever instance
        """
        # Get retrieval parameters from config
        search_k = self.config.get("retriever_k", 3)
        search_type = self.config.get("search_type", "similarity")
        
        logger.info(f"Creating retriever with k={search_k}, type={search_type}")
        
        # Configure retriever
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": search_k},
            search_type=search_type
        )
        
        return retriever