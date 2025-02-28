from langchain.chains import ConversationalRetrievalChain
from src.config.config import get_config
from src.config.logging import get_logger
from typing import Dict, Any

logger = get_logger(__name__)

class ChatChain:
    """
    Build and manage the RAG conversation chain.
    """
    
    def __init__(self, llm, retriever, memory):
        """
        Initialize with required components.
        
        Args:
            llm: Language model
            retriever: Document retriever
            memory: Conversation memory
        """
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.config = get_config()
        
    def create_chain(self):
        """
        Create the conversational retrieval chain.
        
        Returns:
            ConversationalRetrievalChain instance
        """
        logger.info("Creating conversational retrieval chain")
        
        # Get chain parameters from config
        return_source_docs = self.config.get("return_source_docs", True)
        
        # Create the chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=return_source_docs
        )
        
        return qa_chain
    
    @staticmethod
    def format_response(chain_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the chain response for presentation.
        
        Args:
            chain_response: Raw response from the chain
            
        Returns:
            Formatted response
        """
        formatted = {
            "answer": chain_response["answer"],
            "sources": []
        }
        
        # Format source documents
        if "source_documents" in chain_response:
            for i, doc in enumerate(chain_response["source_documents"]):
                formatted["sources"].append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content[:200] + "...",  # Preview
                })
        
        return formatted