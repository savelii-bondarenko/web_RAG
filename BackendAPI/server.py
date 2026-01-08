import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, ConfigDict

from RAG_API import prepare_rag_assets, RAGGraph

app = FastAPI()

rag_app_instance: RAGGraph | None = None

class UserMessage(BaseModel):
    """
    Schema for a user chat message.

    Attributes:
        message (str): The user's input message.
    """
    message: str
    model_config = ConfigDict(extra='forbid')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
    global rag_app_instance
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp_path = tmp.name

    try:
        splitted_chunks, embedder, vector_db = prepare_rag_assets(tmp_path)
        rag_app_instance = RAGGraph(splitted_chunks, embedder, vector_db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")
    finally:
        os.unlink(tmp_path)
        pass

    return {"status": "File processed and RAG initialized"}


@app.post("/chat")
def send_message(user_data: UserMessage):
    """
    Send a user message to the RAG bot and return its response.

    Args:
        user_data (UserMessage): Object containing the user's message.

    Returns:
        dict: A dictionary with the bot response in the form
            {"response": str}.

    Raises:
        HTTPException: If no file has been uploaded before querying the bot.
    """
    if rag_app_instance is None:
        raise HTTPException(status_code=400, detail="Firstly upload file /upload")
    result = rag_app_instance.get_query(user_data.message)["text"]
    return {"response": result}