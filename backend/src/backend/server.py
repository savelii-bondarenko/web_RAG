import os
import tempfile
import asyncio
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, ConfigDict

from rag import prepare_rag_assets, RAGGraph
app = FastAPI()

class User(BaseModel):
    message_history: list[str]
    model_config = ConfigDict(extra='forbid')

rag_sessions: dict[str, tuple[RAGGraph, User]] = {}

class UserMessage(BaseModel):
    """
    Schema for a user chat message.

    Attributes:
        session_id (str): The user's session_id.
    """
    session_id: str
    message: str
    model_config = ConfigDict(extra='forbid')




def generate_id() -> str:
    """Generate a unique ID for a user.

    Returns:
        str: Unique ID.

    """
    return str(uuid.uuid4())

def get_answer_sync(rag_instance, msg):
    return rag_instance.get_query(msg)["text"]

def initialize_rag_sync(path):
    chunks, embedder, vector_db = prepare_rag_assets(path)
    return RAGGraph(chunks, embedder, vector_db)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(
        session_id: str | None = Form(None),
        file: UploadFile = File(...)
):
    """
    Upload a file and initialize the RAG pipeline.

    The uploaded file is temporarily saved to RAM, processed to create
    embeddings and a vector database, and then removed.

    Args:
        file (UploadFile): File uploaded by the user.

    Returns:
        dict: Status message confirming successful initialization {"status": str}.

    Raises:
        HTTPException: If an error occurs during file processing.
    """
    if session_id not in rag_sessions:
        session_id: str = generate_id()
    else:
        del rag_sessions[session_id]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        new_rag_instance: RAGGraph = await asyncio.to_thread(initialize_rag_sync, tmp_path)
        new_user = User(message_history=[])
        rag_sessions[session_id] = (new_rag_instance, new_user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")
    finally:
        os.unlink(tmp_path)
        pass

    return {"status": "File processed and RAG initialized",
            "session_id": session_id}

@app.post("/chat")
async def send_message(user_data: UserMessage):
    """
    Send a user message to the RAG bot and return its response.

    Args:
        user_data (User): Object containing the user's message.

    Returns:
        dict: A dictionary with the bot response in the form
            {"response": str}.

    Raises:
        HTTPException: If no file has been uploaded before querying the bot.
    """
    current_rag: RAGGraph = rag_sessions[user_data.session_id][0]
    current_user: User = rag_sessions[user_data.session_id][1]

    if current_rag is None:
        raise HTTPException(status_code=400, detail="Firstly upload file /upload")
    current_user.message_history.append(user_data.message)
    message: str = "\n".join(current_user.message_history)
    result = await asyncio.to_thread(get_answer_sync, current_rag, message)
    current_user.message_history.append(result)
    return {"response": result}