from fastapi import FastAPI, File, UploadFile
import shutil
import os

app = FastAPI()

# Create the destination folder if it doesn't exist
UPLOAD_DIR = "uploaded_mp3s"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio/")
async def save_audio(file: UploadFile = File(...)):
    # 1. Define the full path where the file will be saved
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 2. Open a local file and write the uploaded content into it
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {
        "filename": file.filename,
        "saved_at": file_path,
        "content_type": file.content_type
    }

@app.get("/hello/")
def read_root():
    return {"Hello": "World"}