import os
import json
from dotenv import load_dotenv

load_dotenv()

_config = None

def load_config():
    """
    Load configuration from config file and environment variables.
    
    Returns:
        Dict containing configuration
    """
    global _config
    
    if _config is not None:
        return _config
    
    config = {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "load_in_8bit": True,
        "use_4bit": False,
        "use_gpu": True,
        
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.15,
        
        "chunk_size": 1000,
        "chunk_overlap": 200,
        
        "retriever_k": 3,
        "search_type": "similarity",
        
        "memory_key": "chat_history",
        "return_messages": True,
        "return_source_docs": True,
        
        "pdf_dir": "./data/pdf_specs",
        "vector_store_dir": "./data/vector_store",
        "model_cache_dir": "./models",
        
        # Logging
        "log_level": "INFO"
    }
    
    config_path = os.environ.get("CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            file_config = json.load(f)
            config.update(file_config)
    
    for key in config.keys():
        env_key = f"PDF_BOT_{key.upper()}"
        if env_key in os.environ:
            try:
                config[key] = json.loads(os.environ[env_key])
            except json.JSONDecodeError:
                config[key] = os.environ[env_key]
    
    _config = config
    return config

def get_config():
    """
    Get the current configuration.
    
    Returns:
        Dict containing configuration
    """
    global _config
    
    if _config is None:
        _config = load_config()
    
    return _config