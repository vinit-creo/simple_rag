import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config.config import get_config
from src.config.logging import get_logger

logger = get_logger(__name__)

class LlamaLoader:
    """
    Load and configure Llama models.
    """
    
    def __init__(self):
        """Initialize with configuration."""
        self.config = get_config()
        
    def load_model_and_tokenizer(self):
        """
        Load the Llama model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        
        
        if torch.backend.mps.is_available():
            device - torch.device("mps")
        else:
            device = torch.device("cpu")
            
        print("MPS not available, using CPU")
        model_name = self.config.get("model_name", "meta-llama/Llama-2-7b-chat-hf")
        load_in_8bit = self.config.get("load_in_8bit", True)
        use_4bit = self.config.get("use_4bit", False)
        
        logger.info(f"Loading Llama model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        device_map = "auto"
        
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization")
            except ImportError:
                logger.warning("BitsAndBytes not available, falling back to 8-bit")
                load_in_8bit = True
                quantization_config = None
        else:
            quantization_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            load_in_8bit=load_in_8bit if not use_4bit else False,
            quantization_config=quantization_config
        )
        
        logger.info(f"Successfully loaded {model_name}")
        
        return model, tokenizer