import os
import shutil
from fastapi import UploadFile
import tempfile

# saving all upload files in this particular folder
UPLOAD_DIR = "./uploaded_docs"

def save_uploaded_files(uploaded_files: list[UploadFile]) -> list[str]:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    saved_file_paths = []
    
    for file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_file_paths.append(save_path)
    
    return saved_file_paths