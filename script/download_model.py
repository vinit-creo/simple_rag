import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.config import get_config
from src.config.logging import setup_logging, get_logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

def main():
    """
    Download and cache the required models.
    """
    # Set up logging
    setup_logging()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download and cache models")
    parser.add_argument("--llm_only", action="store_true", help="Download only the LLM")
    parser.add_argument("--embeddings_only", action="store_true", help="Download only the embedding model")
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    
    # Create cache directory if it doesn't exist
    os.makedirs(config["model_cache_dir"], exist_ok=True)
    
    try:
        # Download LLM
        if not args.embeddings_only:
            logger.info(f"Downloading LLM: {config['model_name']}")
            print(f"Downloading LLM: {config['model_name']}...")
            
            # Set cache directory
            os.environ["TRANSFORMERS_CACHE"] = config["model_cache_dir"]
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
            
            # Download model files (without loading the full model)
            # This will cache the model files for later use
            AutoModelForCausalLM.from_pretrained(
                config["model_name"], 
                torch_dtype="auto",
                device_map=None
            )
            
            logger.info(f"LLM downloaded and cached in {config['model_cache_dir']}")
            print(f"LLM downloaded and cached.")
        
        # Download embedding model
        if not args.llm_only:
            logger.info(f"Downloading embedding model: {config['embedding_model']}")
            print(f"Downloading embedding model: {config['embedding_model']}...")
            
            # Download embedding model
            SentenceTransformer(config["embedding_model"])
            
            logger.info(f"Embedding model downloaded and cached")
            print(f"Embedding model downloaded and cached.")
        
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()