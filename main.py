import src.graph as graph

from fastapi import FastAPI, File, UploadFile
import os

app = FastAPI()

# Define a Pydantic model to accept the question in JSON format
class QuestionRequest(BaseModel):
    question: str

# Endpoint to upload the file
@app.post("/upload/")
async def upload_file():

    return {"message": f"File '{file.filename}' uploaded successfully!", "file_location": file_location}

