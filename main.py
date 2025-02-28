import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config.logging import setup_logging, get_logger
from src.config.config import get_config
from ui.cli import CliInterface
from ui.web import WebInterface

logger = get_logger(__name__)

def main():
    """Main entry point for the PDF Specification Chatbot application."""

    setup_logging()
    
    parser = argparse.ArgumentParser(description="PDF Specification Chatbot")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store")
    parser.add_argument("--pdf_dir", type=str, help="Custom PDF directory")
    args = parser.parse_args()
    
    config = get_config()
    
    if args.pdf_dir:
        config["pdf_dir"] = args.pdf_dir
        logger.info(f"Using custom PDF directory: {args.pdf_dir}")
    
    os.makedirs(config["pdf_dir"], exist_ok=True)
    os.makedirs(config["vector_store_dir"], exist_ok=True)
    os.makedirs(config["model_cache_dir"], exist_ok=True)
    
    pdf_files = [f for f in os.listdir(config["pdf_dir"]) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {config['pdf_dir']}")
        print(f"Warning: No PDF files found in {config['pdf_dir']}")
        print(f"Please add PDF files to {config['pdf_dir']} before building the vector store.")
    
    try:
        if args.web:
            logger.info("Starting web interface")
            web_ui = WebInterface()
            web_ui.setup(rebuild_vector_store=args.rebuild)
            web_ui.launch()
        else:
            logger.info("Starting CLI interface")
            cli_ui = CliInterface()
            cli_ui.run()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        print("\nApplication terminated by user")
    except Exception as e:
        logger.error(f"Error in main application: {str(e)}")
        print(f"\nError: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())