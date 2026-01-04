from langchain_core.messages import AnyMessage, AIMessage
from langgraph.graph import StateGraph


from langchain_core.documents import Document
from numpy import ndarray
from faiss import Index

from typing import TypedDict
from langchain_deepseek import ChatDeepSeek

from dotenv import load_dotenv
load_dotenv()


model = ChatDeepSeek(
    model="deepseek-chat",
    api_key="",
)

class State(TypedDict):
    message: AnyMessage
    answer: dict
    extracted_docs: list[Document]

class RAGGraph:
    def __init__(self,
                 model,
                 splitted_text: list[Document],
                 embedder,
                 vector_db: Index):
        self.model = model
        self.splitted_text = splitted_text
        self.embedder = embedder
        self.vector_db = vector_db
        self.app = self._build_graph()

    def _retriever_node(self, state: State) -> State:
        message: list[Document] = [Document(page_content=state["message"].content)]
        message_embeddings: ndarray = self.embedder.make_embeddings(message)
        _, indices = self.vector_db.search(x=message_embeddings, k=3)
        state["extracted_docs"].extend([self.splitted_text[vec] for vec in indices[0]])
        return state

    def _generate_node(self, state: State) -> State:
        extracted_docs = "\n".join(doc.page_content for doc in state["extracted_docs"])
        sources = list(set(doc.metadata.get("source", "unknown") for doc in state["extracted_docs"]))
        context: str = f"context: {extracted_docs}, question: {state["message"].content}"
        response: AIMessage = self.model.invoke(context)
        state["answer"] = dict(text=response.content, source=sources)
        return state

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("retriever", self._retriever_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "generate")
        graph.set_finish_point("generate")

        return graph.compile()