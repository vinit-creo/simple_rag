from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from src.config.config import get_config
from src.config.logging import get_logger

logger = get_logger(__name__)

class LlamaPipeline:
    """
    Create inference pipeline for Llama models.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize with model and tokenizer.
        
        Args:
            model: Loaded model
            tokenizer: Loaded tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = get_config()
        
    def create_langchain_pipeline(self):
        """
        Create a LangChain compatible pipeline.
        
        Returns:
            HuggingFacePipeline instance
        """
        # Get generation parameters from config
        max_length = self.config.get("max_length", 512)
        temperature = self.config.get("temperature", 0.7)
        top_p = self.config.get("top_p", 0.95)
        repetition_penalty = self.config.get("repetition_penalty", 1.15)
        
        logger.info("Creating LLM pipeline")
        
        # Create text generation pipeline
        text_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Create LangChain wrapper
        llm = HuggingFacePipeline(pipeline=text_pipeline)
        
        return llm