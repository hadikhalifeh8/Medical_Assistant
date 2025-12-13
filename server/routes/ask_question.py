from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

# Note: Imports for `pinecone`, `langchain` and `langchain_google_genai`
# are performed inside the endpoint to avoid crashing the app at
# import-time when incompatible versions are installed. The code below
# provides local fallbacks for the `Document` shape and a simple
# retriever implementing `get_relevant_documents` expected by
# RetrievalQA.

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f'user query: {question}')

        # Lazy imports to avoid import-time failures when packages
        # are missing or incompatible in the environment.
        try:
            from pinecone import Pinecone
        except ModuleNotFoundError:
            return JSONResponse(
                status_code=500,
                content={"error": "pinecone client not installed. Run: pip install pinecone-client"},
            )

        try:
            # from langchain_google_genai import GoogleGenerativeAIEmbeddings
            from langchain_huggingface import HuggingFaceEmbeddings
        except ModuleNotFoundError:
            return JSONResponse(
                status_code=500,
                content={"error": "langchain-google-genai not installed. Run: pip install langchain-google-genai"},
            )

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        index = pc.Index(os.environ.get("PINECONE_INDEX_NAME"))
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

        # Fallback lightweight Document shape used by RetrievalQA: any
        # object with `page_content` and `metadata` is acceptable.
        class SimpleDocument:
            def __init__(self, page_content: str, metadata: dict):
                self.page_content = page_content
                self.metadata = metadata

        docs = [
            SimpleDocument(
                page_content=match.get("metadata", {}).get("text", ""),
                metadata=match.get("metadata", {}),
            )
            for match in res.get("matches", [])
        ]

        # Simple retriever that exposes `get_relevant_documents` for
        # RetrievalQA. When possible, subclass the langchain `BaseRetriever`
        # so pydantic validation in RetrievalQA accepts the instance. We
        # try several import paths to support different langchain layouts.
        BaseRetriever = None
        try:
            from langchain.schema import BaseRetriever as _BR
            BaseRetriever = _BR
        except Exception:
            try:
                from langchain_core.schema import BaseRetriever as _BR
                BaseRetriever = _BR
            except Exception:
                BaseRetriever = None

        if BaseRetriever is not None:
            class SimpleRetriever(BaseRetriever):
                tags: Optional[List[str]] = Field(default_factory=list)
                metadata: Optional[dict] = Field(default_factory=dict)

                def __init__(self, documents: List[SimpleDocument]):
                    super().__init__()
                    self._docs = documents

                def get_relevant_documents(self, query: str) -> List[SimpleDocument]:
                    return self._docs
        else:
            class SimpleRetriever:
                tags: Optional[List[str]] = Field(default_factory=list)
                metadata: Optional[dict] = Field(default_factory=dict)

                def __init__(self, documents: List[SimpleDocument]):
                    self._docs = documents

                def get_relevant_documents(self, query: str) -> List[SimpleDocument]:
                    return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})