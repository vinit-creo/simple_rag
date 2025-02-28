import gradio as gr
from src.data.pdf_processor import PDFProcessor
from src.data.text_splitter import DocumentSplitter
from src.embeddings.embeddor import Embedder
from src.embeddings.vector_store import VectorStore
from src.llm.load_models import LlamaLoader
from src.llm.pipeline import LlamaPipeline
from src.retrival import Retriever
from src.chat.memory import ConversationMemory
from src.chat.chain import ChatChain
from src.config.config import get_config
from src.config.config import get_logger, setup_logging
import os

logger = get_logger(__name__)

class WebInterface:
    """
    Web interface for the PDF chatbot using Gradio.
    """
    
    def __init__(self):
        """Initialize the web interface."""
        setup_logging()
        self.config = get_config()
        self.chain = None
        
    def setup(self, rebuild_vector_store=False, pdf_dir=None):
        """
        Set up the chatbot components.
        
        Args:
            rebuild_vector_store: Whether to rebuild the vector store
            pdf_dir: Custom PDF directory
            
        Returns:
            ChatChain instance
        """
        logger.info("Setting up PDF chatbot for web interface")
        
        # Override PDF directory if provided
        if pdf_dir:
            self.config["pdf_dir"] = pdf_dir
        
        # Set up embedding model
        embedder = Embedder()
        embedding_model = embedder.get_embedder()
        
        # Set up vector store
        vector_store_manager = VectorStore(embedding_model)
        
        # Check if we need to rebuild the vector store
        if rebuild_vector_store:
            logger.info("Rebuilding vector store")
            # Process PDFs
            pdf_processor = PDFProcessor(self.config["pdf_dir"])
            documents = pdf_processor.process_all_pdfs()
            
            # Split documents
            doc_splitter = DocumentSplitter(
                chunk_size=self.config["chunk_size"],
                chunk_overlap=self.config["chunk_overlap"]
            )
            chunks = doc_splitter.split_documents(documents)
            
            # Create vector store
            vector_store = vector_store_manager.create_from_documents(chunks)
        else:
            # Load existing vector store
            vector_store = vector_store_manager.load()
        
        # Set up model
        model_loader = LlamaLoader()
        model, tokenizer = model_loader.load_model_and_tokenizer()
        
        # Set up pipeline
        llm_pipeline = LlamaPipeline(model, tokenizer)
        llm = llm_pipeline.create_langchain_pipeline()
        
        # Set up retriever
        retriever_manager = Retriever(vector_store)
        retriever = retriever_manager.get_retriever()
        
        # Set up memory
        memory_manager = ConversationMemory()
        memory = memory_manager.create_memory()
        
        # Set up chain
        chain_manager = ChatChain(llm, retriever, memory)
        chain = chain_manager.create_chain()
        
        self.chain = chain
        return chain
    
    def process_query(self, message, history):
        """
        Process a query and return the response.
        
        Args:
            message: User query
            history: Chat history
            
        Returns:
            Bot response
        """
        if not self.chain:
            return "Error: Chatbot not initialized properly. Please check logs."
        
        try:
            # Get response from chain
            response = self.chain({"question": message})
            
            # Format response
            formatted = ChatChain.format_response(response)
            
            # Format sources for display
            source_text = ""
            if formatted["sources"]:
                source_text = "\n\n**Sources:**\n"
                for i, source in enumerate(formatted["sources"]):
                    source_text += f"- {source['source']}\n"
            
            # Combine answer and sources
            full_response = formatted["answer"] + source_text
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in web chat: {str(e)}")
            return "An error occurred while processing your query. Please try again."
    
    def get_pdf_files(self):
        """Get list of PDF files in the configured directory."""
        try:
            pdf_files = [f for f in os.listdir(self.config["pdf_dir"]) if f.lower().endswith('.pdf')]
            return pdf_files
        except Exception as e:
            logger.error(f"Error listing PDF files: {str(e)}")
            return []
    
    def launch(self):
        """Launch the Gradio web interface."""
        # Get PDF files
        pdf_files = self.get_pdf_files()
        pdf_list = "\n".join(pdf_files) if pdf_files else "No PDF files found"
        
        # Set up the Gradio interface
        with gr.Blocks(title="PDF Specification Chatbot") as demo:
            gr.Markdown("# PDF Specification Chatbot")
            gr.Markdown("Ask questions about your technical specifications and documents.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(label="Ask a question", placeholder="What does the specification say about...", lines=2)
                    
                    with gr.Row():
                        submit_btn = gr.Button("Submit")
                        clear_btn = gr.Button("Clear Chat")
                
                with gr.Column(scale=1):
                    gr.Markdown("## Available Documents")
                    docs_display = gr.Textbox(value=pdf_list, label="PDF Files", interactive=False, lines=10)
                    rebuild_btn = gr.Button("Rebuild Knowledge Base")
            
            # Set up interactions
            submit_btn.click(self.process_query, inputs=[msg, chatbot], outputs=[chatbot])
            msg.submit(self.process_query, inputs=[msg, chatbot], outputs=[chatbot])
            clear_btn.click(lambda: None, None, chatbot, queue=False)
            
            # Rebuild knowledge base
            rebuild_btn.click(
                lambda: self.setup(rebuild_vector_store=True),
                inputs=[],
                outputs=[],
                queue=True
            )
        
        # Launch the interface
        demo.launch(share=True)