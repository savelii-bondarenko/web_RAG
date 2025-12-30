from utills import read_data, split_text, Embedder, create_vectorDB
from langchain_core.documents import Document
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

data = "1.docx"

#read data
extracted_text: str = read_data(data)

#split text
splitted_text: list[Document] = split_text(extracted_text)

#create embeddings
embeddings = Embedder().make_embeddings(splitted_text)

#create vectorDB
vectorDB = create_vectorDB(embeddings)





