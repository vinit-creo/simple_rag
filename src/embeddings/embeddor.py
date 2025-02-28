import langchain_huggingface


langchain_huggingface.HuggingFaceEmbeddings
from src.config.config import get_config
from src.config.logging import get_logger

logger = get_logger(__name__)

class Embedder:
    """
    Create text embeddings for document chunks.
    """
    
    def __init__(self):
        """Initialize the embedder with the appropriate model."""
        config = get_config()
        self.model_name = config.get(
            "embedding_model", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        logger.info(f"Initializing embedder with model: {self.model_name}")
        
        self.embeddings = langchain_huggingface.HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cuda" if config.get("use_gpu", False) else "cpu"}
        )
    
    def get_embedder(self):
        """Get the embeddings model instance."""
        return self.embeddings