import argparse
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.pdf_processor import PDFProcessor
from src.data.text_splitter import DocumentSplitter
from src.embeddings.embeddor import Embedder
from src.embeddings.vector_store import VectorStore
from src.config.config import get_config
from src.config.logging import setup_logging, get_logger

logger = get_logger(__name__)

def main():
    """
    Build the vector store from PDF files.
    """
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Build vector store from PDF files")
    parser.add_argument("--pdf_dir", type=str, help="PDF directory")
    parser.add_argument("--vector_dir", type=str, help="Vector store directory")
    parser.add_argument("--chunk_size", type=int, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, help="Chunk overlap")
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    
    # Override config with command line arguments
    if args.pdf_dir:
        config["pdf_dir"] = args.pdf_dir
    if args.vector_dir:
        config["vector_store_dir"] = args.vector_dir
    if args.chunk_size:
        config["chunk_size"] = args.chunk_size
    if args.chunk_overlap:
        config["chunk_overlap"] = args.chunk_overlap
    
    # Check if PDF directory exists
    if not os.path.exists(config["pdf_dir"]):
        logger.error(f"PDF directory not found: {config['pdf_dir']}")
        print(f"Error: PDF directory not found: {config['pdf_dir']}")
        return
    
    logger.info(f"Building vector store from PDFs in {config['pdf_dir']}")
    logger.info(f"Using chunk size {config['chunk_size']} and overlap {config['chunk_overlap']}")
    
    try:
        # Process PDFs
        pdf_processor = PDFProcessor(config["pdf_dir"])
        documents = pdf_processor.process_all_pdfs()
        logger.info(f"Processed {len(documents)} PDF documents")
        
        # Split documents
        doc_splitter = DocumentSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )
        chunks = doc_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        
        # Create embeddings and vector store
        embedder = Embedder()
        embedding_model = embedder.get_embedder()
        
        vector_store_manager = VectorStore(embedding_model)
        vector_store = vector_store_manager.create_from_documents(chunks)
        
        logger.info(f"Vector store created and saved to {config['vector_store_dir']}")
        print(f"Success: Vector store created with {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error building vector store: {str(e)}")
        print(f"Error: {str(e)}")
        
if __name__ == "__main__":
    main()