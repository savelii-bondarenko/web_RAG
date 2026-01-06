import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, ConfigDict

from RAG_API import prepare_rag_assets, RAGGraph

app = FastAPI()

rag_app_instance: RAGGraph | None = None

class UserMessage(BaseModel):
    message: str
    model_config = ConfigDict(extra='forbid')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
    if rag_app_instance is None:
        raise HTTPException(status_code=400, detail="Сначала загрузите файл через /upload")
    result = rag_app_instance.get_query(user_data.message)["text"]
    return {"response": result}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)