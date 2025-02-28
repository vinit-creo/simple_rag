from setuptools import setup, find_packages

setup(
    name="pdf_spec_chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.267",
        "transformers>=4.30.2",
        "torch>=2.0.1",
        "pymupdf>=1.22.3",
        "sentence-transformers>=2.2.2",
        "chromadb>=0.4.6",
        "peft>=0.4.0",
        "accelerate>=0.21.0",
        "bitsandbytes>=0.40.2",
        "gradio>=3.38.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'pdf-chatbot=main:main',
        ],
    },
    author="vinit",
    author_email="vinitmp.work@gmail.com",
    description="A chatbot for querying PDF technical specifications using Llama",
    keywords="pdf, llm, chatbot, llama, rag",

)