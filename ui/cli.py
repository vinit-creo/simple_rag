import argparse
from src.data.pdf_processor import PDFProcessor
from src.data.text_splitter import DocumentSplitter
from src.embeddings.embeddor import Embedder
from src.embeddings.vector_store import VectorStore
from src.llm.load_models import LlamaLoader
from src.llm.pipeline import LlamaPipeline
from src.retrival.retrival import Retriever
from src.chat.memory import ConversationMemory
from src.chat.chain import ChatChain
from src.config.config import get_config
from src.config.logging import get_logger, setup_logging

logger = get_logger(__name__)

class CliInterface:
    """
    Command line interface for the PDF chatbot.
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        setup_logging()
        self.config = get_config()
        
    def setup(self, rebuild_vector_store=False):
        """
        Set up the chatbot components.
        
        Args:
            rebuild_vector_store: Whether to rebuild the vector store
            
        Returns:
            ChatChain instance
        """
        logger.info("Setting up PDF chatbot")
        
        embedder = Embedder()
        embedding_model = embedder.get_embedder()
        
        vector_store_manager = VectorStore(embedding_model)
        
        if rebuild_vector_store:
            logger.info("Rebuilding vector store")
            pdf_processor = PDFProcessor(self.config["pdf_dir"])
            documents = pdf_processor.process_all_pdfs()
            
            doc_splitter = DocumentSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            chunks = doc_splitter.split_documents(documents)
            
            vector_store = vector_store_manager.create_from_documents(chunks)
        else:
            vector_store = vector_store_manager.load()
        
        model_loader = LlamaLoader()
        model, tokenizer = model_loader.load_model_and_tokenizer()
        
        llm_pipeline = LlamaPipeline(model, tokenizer)
        llm = llm_pipeline.create_langchain_pipeline()
        
        retriever_manager = Retriever(vector_store)
        retriever = retriever_manager.get_retriever()
        
        memory_manager = ConversationMemory()
        memory = memory_manager.create_memory()
        
        chain_manager = ChatChain(llm, retriever, memory)
        chain = chain_manager.create_chain()
        
        return chain
    
    def run(self):
        """Run the CLI interface."""
        parser = argparse.ArgumentParser(description="PDF Specification Chatbot CLI")
        parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store")
        args = parser.parse_args()
        
        # Set up chain
        chain = self.setup(rebuild_vector_store=args.rebuild)
        
        # Run chat loop
        print("\n==== PDF Specification Chatbot ====")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("Type 'clear' to clear conversation history.")
        print("====================================\n")
        
        while True:
            query = input("\nYou: ")
            
            if query.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break
            
            if query.lower() == "clear":
                chain.memory.clear()
                print("\nConversation history cleared.")
                continue
            
            try:
                response = chain({"question": query})
                formatted = ChatChain.format_response(response)
                
                print("\nBot:", formatted["answer"])
                
                if formatted["sources"]:
                    print("\nSources:")
                    for i, source in enumerate(formatted["sources"]):
                        print(f"  {i+1}. {source['source']}")
                
            except Exception as e:
                logger.error(f"Error in chat: {str(e)}")
                print("\nAn error occurred while processing your query. Please try again.")