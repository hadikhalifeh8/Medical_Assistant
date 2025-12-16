from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from modules.hf_embeddings import embed_text
from pydantic import Field
from typing import List, Optional
from logger import logger
import os

router = APIRouter()

@router.post("/ask/")
async def ask_question(question: str = Form(...)):
    try:
        logger.info(f"User query: {question}")

        from pinecone import Pinecone
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", ""))
        index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "medical-index"))

        embedded_query = embed_text(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)

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

        # Pydantic v2 compatible retriever
        BaseRetriever = None
        try:
            from langchain.schema import BaseRetriever as _BR
            BaseRetriever = _BR
        except:
            pass

        if BaseRetriever:
            class SimpleRetriever(BaseRetriever):
                tags: List[str] = Field(default_factory=list)
                metadata: dict = Field(default_factory=dict)

                def __init__(self, documents: List[SimpleDocument]):
                    super().__init__()
                    self._docs = documents

                def get_relevant_documents(self, query: str) -> List[SimpleDocument]:
                    return self._docs
        else:
            class SimpleRetriever:
                tags: List[str] = Field(default_factory=list)
                metadata: dict = Field(default_factory=dict)

                def __init__(self, documents: List[SimpleDocument]):
                    self._docs = documents

                def get_relevant_documents(self, query: str) -> List[SimpleDocument]:
                    return self._docs

        retriever = SimpleRetriever(docs)
        chain = get_llm_chain(retriever)
        result = query_chain(chain, question)

        logger.info("Query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})
