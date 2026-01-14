from numpy import ndarray

from .utills import (read_data, split_text, Embedder, create_vectorDB)
from langchain_core.documents import Document
from faiss import Index

def prepare_rag_assets(file_path: str):
    """Prepare data for RAG

    Args:
        file_path (str): Path to the file

    Returns:
        splitted_text (str): splitted text for chunks.

        embedder (Embedder): embedder class for create embeddings.

        vector_db (faiss.Index): vector database.
    """
    extracted_text: str = read_data(file_path)
    splitted_text: list[Document] = split_text(extracted_text)
    embedder = Embedder()
    embeddings: ndarray = embedder.make_embeddings(splitted_text)
    vector_db: Index = create_vectorDB(embeddings)

    return splitted_text, embedder, vector_db