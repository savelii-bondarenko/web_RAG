import logging

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph import StateGraph
from sqlalchemy.testing.suite.test_reflection import metadata

from utills import read_data, split_text, Embedder, create_vectorDB
from langchain_core.documents import Document
from numpy import ndarray
from faiss import Index

from typing import TypedDict
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

data = "1.pdf"

extracted_text: str = read_data(data)
splitted_text: list[Document] = split_text(extracted_text)
embedder: Embedder = Embedder()
embeddings: ndarray = embedder.make_embeddings(splitted_text)
vectorDB: Index = create_vectorDB(embeddings)

model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="",
)

class State(TypedDict):
    message: AnyMessage
    answer: AnyMessage
    extracted_docs: list[Document]

def retriever_node(state: State) -> State:
    message: list[Document] = [Document(page_content=state["message"].content)]
    message_embeddings: ndarray = embedder.make_embeddings(message)
    _, indices = vectorDB.search(x=message_embeddings, k=3)
    state["extracted_docs"].extend([splitted_text[vec] for vec in indices[0]])
    return state

def generate_node(state: State) -> State:
    extracted_docs = "\n".join(doc.page_content for doc in state["extracted_docs"])
    context = f"context: {extracted_docs}, question: {state["message"].content}"
    state["answer"] = model.invoke(context)
    return state

graph = StateGraph(State)

graph.add_node("retriever", retriever_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retriever")
graph.add_edge("retriever", "generate")
graph.set_finish_point("generate")

app = graph.compile()

some = app.invoke({
    "message": HumanMessage(content="Привет, что такое философский камень?"),
    "extracted_docs": [],
    "answer": ""
})
print(some["answer"].content)













