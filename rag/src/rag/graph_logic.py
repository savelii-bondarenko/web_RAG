import os

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
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
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

class State(TypedDict):
    message: AnyMessage
    answer: dict
    extracted_docs: list[Document]

class RAGGraph:
    """ Initializes the RAGGraph.

    Args:
        splitted_text (list[Document]): List of pre-split source documents
            corresponding to the embeddings stored in the vector database.
        embedder: Embedding model with a `make_embeddings` method that
            converts documents into numpy arrays.
        vector_db (Index): FAISS index used for similarity search.
    """
    def __init__(self,
                 splitted_text: list[Document],
                 embedder,
                 vector_db: Index):

        self.model = model
        self.splitted_text = splitted_text
        self.embedder = embedder
        self.vector_db = vector_db

    def _retriever_node(self, state: State) -> State:
        """
        Retriever LangGraph node.

        Generates an embedding for the user query, performs a similarity
        search in the vector database, and appends the top-k retrieved
        documents to the graph state.

        Side Effects:
            Mutates `state["extracted_docs"]` by extending it with retrieved
            documents.

        Args:
            state (State): Current graph state. Expects:
                - state["message"] to contain the user query.
                - state["extracted_docs"] to be initialized as a list.

        Returns:
            State: Updated state with retrieved documents appended.
        """
        message: list[Document] = [Document(page_content=state["message"].content)]
        message_embeddings: ndarray = self.embedder.make_embeddings(message)
        _, indices = self.vector_db.search(x=message_embeddings, k=3)
        state["extracted_docs"].extend([self.splitted_text[vec] for vec in indices[0]])
        return state

    def _generate_node(self, state: State) -> State:
        """
        Generation LangGraph node.

        Builds a context from retrieved documents and uses the language
        model to generate a final answer. The generated answer and
        document sources are stored in the graph state.

        Side Effects:
            Sets `state["answer"]` with generated text and source metadata.

        Args:
            state (State): Current graph state. Expects:
                - state["message"] to contain the user query.
                - state["extracted_docs"] to contain retrieved documents.

        Returns:
            State: Updated state with the generated answer.
        """
        extracted_docs = "\n".join(doc.page_content for doc in state["extracted_docs"])
        sources = list(set(doc.metadata.get("source", "unknown") for doc in state["extracted_docs"]))
        context: str = f"context: {extracted_docs}, question: {state["message"].content}"
        response: AIMessage = self.model.invoke(context)
        state["answer"] = dict(text=response.content, source=sources)
        return state

    def _build_graph(self):
        """
        Builds and compiles the LangGraph state graph.

        The graph has the following structure:
            entry -> retriever -> generate -> finish

        Returns:
            Compiled LangGraph application.
        """
        graph = StateGraph(State)

        graph.add_node("retriever", self._retriever_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("retriever")
        graph.add_edge("retriever", "generate")
        graph.set_finish_point("generate")

        return graph.compile()

    def get_query(self, user_question: str):
        """
        Send and get text to and from LLM
        Args:
            user_question (str): User's query.

        Returns:
            result["answer"] (dict(text, source)): Answer of the query.

        """
        app = self._build_graph()
        initial_state = {
            "message": HumanMessage(content=user_question),
            "answer": {},
            "extracted_docs": []
        }
        result = app.invoke(initial_state)
        return result["answer"]
