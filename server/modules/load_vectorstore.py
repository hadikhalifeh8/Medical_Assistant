import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
"""Pinecone is imported lazily via get_pinecone_index() so the app
can start even when the environment has an unexpected Pinecone
installation. If the import fails with the rename/packaging error,
we surface a clear message instructing the developer how to fix the
environment.
"""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medical-index"


os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

_pinecone_index = None


def get_pinecone_index():
    """Lazily import and return a ready Pinecone index instance.

    Raises RuntimeError with actionable instructions if a packaging
    conflict is detected (e.g. both `pinecone-client` and `pinecone`
    installed or an incorrect package present).
    """
    global _pinecone_index
    if _pinecone_index is not None:
        return _pinecone_index

    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as e:
        # Surface the packaging rename message as a runtime error with
        # remediation steps instead of letting import-time crash the
        # whole application.
        msg = str(e)
        hint = (
            "Pinecone import failed. Ensure the official package `pinecone` "
            "is installed and remove any legacy `pinecone-client` package.\n"
            "Run: pip uninstall -y pinecone-client pinecone; pip install pinecone"
        )
        raise RuntimeError(hint + "\n\nOriginal error:\n" + msg) from e

    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=spec,
        )
    else:
        index_info = pc.describe_index(PINECONE_INDEX_NAME)
        try:
            dim = index_info.dimension
        except Exception:
            dim = None
        if dim and dim != 384:
            print(f"Index dimension mismatch: {dim} != 384. Deleting and recreating...")
            pc.delete_index(PINECONE_INDEX_NAME)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,
                metric="dotproduct",
                spec=spec,
            )

    # Wait until ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

    _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return _pinecone_index

# load, split, embed, and upsert pdfs docs content

def load_vectorstore(uploaded_files):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    file_paths = []
    
    # 1- upload
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
        
    # 2- split
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        
        
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
        
        # 3- Embedding
        print("Embedding Chunks")

        # Attempt embedding with retries for transient failures.
        max_retries = 2
        embeddings = None
        try:
            embeddings = embed_model.embed_documents(texts)
        except Exception as e:
            logger = None
            try:
                from logger import logger as _logger
                logger = _logger
            except Exception:
                pass

            msg = str(e)
            # If the error message indicates quota exhaustion, raise a clear error immediately.
            if "Quota exceeded" in msg or "embed_content_free_tier_requests" in msg or "ResourceExhausted" in msg:
                if logger:
                    logger.error("Embedding failed due to quota limits: %s", msg)
                raise RuntimeError(
                    "Embedding quota exceeded for Gemini API. Check your billing/quotas at https://ai.google.dev/gemini-api/docs/rate-limits and ensure the GEMINI_API_KEY project has billing enabled."
                ) from e

            # Otherwise, retry a few times with exponential backoff.
            for attempt in range(1, max_retries + 1):
                backoff = 2 ** attempt
                if logger:
                    logger.warning("Embedding attempt %d failed, retrying in %ds: %s", attempt, backoff, msg)
                time.sleep(backoff)
                try:
                    embeddings = embed_model.embed_documents(texts)
                    break
                except Exception as e2:
                    msg = str(e2)
                    if attempt == max_retries:
                        if logger:
                            logger.error("Embedding failed after retries: %s", msg)
                        raise RuntimeError("Failed to embed documents: " + msg) from e2

        # 4- upsert
        print("upserting embeddings...")
        pc_index = get_pinecone_index()
        with tqdm(total=len(embeddings), desc="upserting to Pinecone") as progress:
            pc_index.upsert(vectors=zip(ids, embeddings, metadata))
            progress.update(len(embeddings))
            
        print(f"Upload completed for ({file_path})")
        
        
            
    
    
    

