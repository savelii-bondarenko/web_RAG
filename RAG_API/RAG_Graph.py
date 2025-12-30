from utills import read_data, split_text, Embedder, create_vectorDB
from langchain_core.documents import Document
from numpy import ndarray
from faiss import Index
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

data = "1.docx"

extracted_text: str = read_data(data)
splitted_text: list[Document] = split_text(extracted_text)
embeddings: ndarray = Embedder().make_embeddings(splitted_text)
vectorDB: Index = create_vectorDB(embeddings)





