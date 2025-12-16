import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.hf_embeddings import embed_documents
load_dotenv()

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"

_pinecone_index = None

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index:
        return _pinecone_index

    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    if PINECONE_INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="dotproduct", spec=spec)

    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

    _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index

def load_vectorstore(uploaded_files):
    file_paths = []

    # 1- save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    # 2- load, split, embed, upsert
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]

        print("Embedding chunks...")
        embeddings = embed_documents(texts)

        print("Upserting to Pinecone...")
        pc_index = get_pinecone_index()
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            pc_index.upsert(vectors=zip(ids, embeddings, metadata))
            progress.update(len(embeddings))

        print(f"Upload completed for {file_path}")
