import logging
import os
from src.config.config import get_config

# Configure logging
def setup_logging():
    """Configure global logging settings."""
    config = get_config()
    log_level_str = config.get("log_level", "INFO")
    log_level = getattr(logging, log_level_str)
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/pdf_bot.log"),
            logging.StreamHandler()
        ]
    )

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)