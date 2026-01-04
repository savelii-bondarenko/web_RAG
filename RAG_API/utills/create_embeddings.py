import numpy as np
from FlagEmbedding import BGEM3FlagModel
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class Embedder:
    """
    Args:
        model_name (str): HuggingFace model name.
    """
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        try:
            self.model = BGEM3FlagModel(model_name)
            logger.info(f"BGEM3 flag model loaded: {model_name}")
        except Exception as e:
            logger.critical("Failed to load BGEM3 flag model")
            raise e

    def make_embeddings(self,
                        data: list[Document],
                        batch_size: int = 128) -> np.ndarray:
        """Make embeddings from a list of documents.

        Args:
            data (list[Document]): List of documents.
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            np.ndarray: 2D array of dense_vecs.

        Raises:
            Exception: If encoding fails.
        """
        try:
            prepared_data = [line.page_content for line in data]
            embeddings = self.model.encode(
                sentences=prepared_data,
                batch_size=batch_size,
                return_dense=True
            )
            logger.info("Embeddings prepared")
            return embeddings["dense_vecs"]
        except Exception as e:
            logger.critical("Failed to encode documents")
            raise e

