"""Simple script to populate a vector database with PDFs."""

import logging
import os
import shutil
from pathlib import Path

import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

dotenv.load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_and_chunk_pdfs(data_dir: str) -> list[Document]:
    """Load PDFs from directory and split into chunks."""
    logger.info("Loading PDFs from %s", data_dir)
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)


def create_vector_store(chunks: list[Document], persist_directory: str) -> Chroma:
    """Create and persist Chroma vector store."""
    # Clear existing vector store if it exists
    if Path(persist_directory).exists():
        logger.info("Clearing existing vector store at %s", persist_directory)
        shutil.rmtree(persist_directory)

    # Initialize HuggingFace embeddings
    embeddings_engine = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cpu"}
    )

    # Create and persist Chroma vector store
    logger.info("Creating new vector store...")
    return Chroma.from_documents(documents=chunks, embedding=embeddings_engine, persist_directory=persist_directory)


def main() -> None:
    """Erase and repopulate vector database."""
    # Define directories
    data_dir = os.getenv("DATA_DIR")
    db_dir = os.getenv("DB_DIR")

    # Process PDFs
    logger.info("Loading and processing PDFs...")
    chunks = load_and_chunk_pdfs(str(data_dir))
    logger.info("Created %d chunks from PDFs", len(chunks))

    # Create vector store
    logger.info("Creating vector store...")
    vectordb = create_vector_store(chunks, str(db_dir))
    logger.info("Vector store created and persisted at %s", db_dir)
    results = vectordb.similarity_search("What is the best client?", k=1)
    logger.info("Results: %s", results)


if __name__ == "__main__":
    main()
