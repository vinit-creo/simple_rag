from langchain.memory import ConversationBufferMemory
from src.config.config import get_config

class ConversationMemory:
    """
    Manage conversation history.
    """
    
    def __init__(self):
        """Initialize with configuration."""
        self.config = get_config()
        
    def create_memory(self):
        """
        Create a conversation memory instance.
        
        Returns:
            ConversationBufferMemory instance
        """
        memory_key = self.config.get("memory_key", "chat_history")
        return_messages = self.config.get("return_messages", True)
        
        memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages
        )
        
        return memory