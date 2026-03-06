# web_RAG

A Retrieval-Augmented Generation (RAG) agent that enables users to upload documents and interact with them through a chat interface. This project utilizes a microservices architecture with a FastAPI backend, a Streamlit frontend, and a custom RAG engine powered by LangGraph and DeepSeek.

## 🚀 Features

* **Document Ingestion**: Supports uploading and processing of `.txt`, `.pdf`, `.docx`, and `.xlsx` files.
* **Interactive Chat**: User-friendly chat interface with message history preservation.
* **Intelligent RAG Pipeline**:
* Uses **LangGraph** for orchestrating the retrieval and generation flow.
* **DeepSeek** LLM integration for high-quality responses.
* **FAISS** vector database for efficient similarity search.


* **Tool Usage**: The agent is equipped with basic mathematical tools (add, subtract, multiply, divide, min, max, average) to perform calculations during the conversation.
* **Containerized**: Fully Dockerized for easy deployment using Docker Compose.

## 🛠️ Tech Stack

* **Language**: Python 3.12+
* **Package Manager**: `uv`
* **Backend**: FastAPI
* **Frontend**: Streamlit
* **RAG & LLM**: LangChain, LangGraph, DeepSeek API, FAISS
* **Infrastructure**: Docker, Docker Compose

## 📂 Project Structure

The project is organized into three main workspaces:

* **`backend/`**: Contains the FastAPI server logic and API endpoints.
* **`frontend/`**: Contains the Streamlit UI application.
* **`rag/`**: Contains the core logic for the RAG engine, vector store management, and graph definitions.

## ⚙️ Installation & Setup

### Prerequisites

* **Docker** and **Docker Compose**
* (Optional for local dev) **Python 3.12+** and **uv**

### Environment Configuration

Create a `.env` file in the root directory. You will need to configure the vector database search parameters. It is how deeply RAG can search.

```env
VDB_SEARCH_K=5
```

### Running with Docker (Recommended)

1. Build and start the services:
```bash
docker compose build
docker compose up

```


2. Access the applications:
* **Frontend**: http://localhost:8501
* **Backend API**: http://localhost:8000/docs



The `docker-compose.yml` defines two services: `backend` and `frontend`. The backend service initializes the RAG pipeline, and the frontend service provides the user interface.

### Running Locally (Manual)

If you prefer to run services individually without Docker:

1. **Install dependencies**:
Ensure you have `uv` installed, then sync the project.
2. **Start the Backend**:
```bash
uv run fastapi run backend/src/backend/server.py --port 8000 --host 0.0.0.0

```


3. **Start the Frontend**:
In a separate terminal, run:
```bash
export BACKEND_URL="http://localhost:8000"
streamlit run frontend/src/frontend/frontend.py --server.port 8501

```



## 🔌 API Endpoints

The backend exposes the following endpoints:

* `GET /health`: Health check to verify the API is running.
* `POST /upload`: Upload a file to initialize a RAG session.
* **Returns**: `session_id` and status.


* `POST /chat`: Send a message to the RAG agent.
* **Payload**: `{"session_id": "...", "message": "..."}`
* **Returns**: The agent's response.



## 📖 Usage

1. Open the frontend application in your browser (`http://localhost:8501`).
2. Use the sidebar to **Upload a file** (Supported formats: `.txt`, `.pdf`, `.docx`, `.xlsx`).
3. Once the file is uploaded and processed, use the chat input to ask questions about the document.
4. The agent will retrieve relevant context and generate an answer, potentially using math tools if calculation is required.

