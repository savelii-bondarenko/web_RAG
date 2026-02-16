import faiss
import numpy as np
import logging

logger = logging.getLogger(__name__)
def create_vectorDB(embeddings: np.ndarray) -> faiss.Index:
    """Create faiss index from embeddings.

    Args:
        embeddings (np.ndarray): 2D numpy array with shape (n, d).

    Returns:
        faiss.Index: faiss index(vector DB object).

    Raises:
        Exception: If failed to create faiss index.
    """
    try:
        dimension = embeddings.shape[1]
        vectorstore = faiss.IndexFlatIP(dimension)
        vectorstore.add(embeddings.astype(np.float32))
        logger.info("Created faiss vectorstore")
        return vectorstore
    except Exception as e:
        logger.critical("Failed to create vectorDB")
        raise e

