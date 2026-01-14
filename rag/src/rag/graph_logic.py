import os

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.documents import Document
from numpy import ndarray
from faiss import Index

from typing import TypedDict, Annotated
from langchain_deepseek import ChatDeepSeek

from .utills import TOOLS

from dotenv import load_dotenv
load_dotenv()


model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
).bind_tools(TOOLS)

class State(TypedDict):
    extracted_docs: list[Document]
    messages: Annotated[list[AnyMessage], add_messages]

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
        self._tool_node = ToolNode(TOOLS)

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
                - state["messages"] to contain the user query.
                - state["extracted_docs"] to be initialized as a list.

        Returns:
            State: Updated state with retrieved documents appended.
        """
        last_message = state["messages"][-1]

        message_doc = [Document(page_content=last_message.content)]
        message_embeddings: ndarray = self.embedder.make_embeddings(message_doc)
        _, indices = self.vector_db.search(x=message_embeddings, k=int(os.getenv("VDB_SEARCH_K")))
        found_docs: list[Document] = [self.splitted_text[vec] for vec in indices[0]]
        return {"extracted_docs": found_docs}

    def _generate_node(self, state: State) -> State:
        """
        Generation LangGraph node.

        Builds a context from retrieved documents and uses the language
        model to generate a final answer. The generated answer and
        document sources are stored in the graph state.

        Args:
            state (State): Current graph state. Expects:
                - state["messages"] to contain the user query.
                - state["extracted_docs"] to contain retrieved documents.

        Returns:
            State: Updated state with the generated answer.
        """
        docs_content = "\n".join(doc.page_content for doc in state["extracted_docs"])
        system_msg = SystemMessage(
            content=f"Context:\n{docs_content}\n\nAnswer the user's question using the context provided."
        )
        messages = [system_msg] + state["messages"]
        response: AIMessage = self.model.invoke(messages)
        return {"messages": [response]}

    def _should_continue_node(self, state: State) -> str:
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "end"

    def _build_graph(self):
        graph = StateGraph(State)

        graph.add_node("retriever", self._retriever_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("tools", self._tool_node)

        graph.set_entry_point("retriever")

        graph.add_edge("retriever", "generate")

        graph.add_conditional_edges(
            "generate",
            self._should_continue_node,
            {
                "tools": "tools",
                "end": END
            }
        )
        graph.add_edge("tools", "generate")
        return graph.compile()

    def get_query(self, user_question: str):
        """
        Send and get text to and from LLM
        Args:
            user_question (str): User's query.

        Returns:
            dict(text): LLM answer

        """
        app = self._build_graph()
        initial_state = {
            "messages": [HumanMessage(content=user_question)],
            "extracted_docs": []
        }
        result = app.invoke(initial_state)
        last_msg = result["messages"][-1]
        return {
            "text": last_msg.content,
        }
