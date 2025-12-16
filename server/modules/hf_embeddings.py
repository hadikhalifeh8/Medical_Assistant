import os
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ENDPOINT = os.getenv("HUGGINGFACE_ENDPOINT") or os.getenv("HUGGINGFACE_URL")

if HF_ENDPOINT:
    # Hosted inference via HuggingFace InferenceClient
    client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

    def embed_text(text: str) -> list[float]:
        response = client.feature_extraction(text, model="intfloat/e5-small-v2")
        if isinstance(response[0], list):
            return response[0]
        return response

    def embed_documents(docs: list[str]) -> list[list[float]]:
        vectors = []
        for doc in docs:
            vectors.append(embed_text(doc))
        return vectors
else:
    # Local fallback
    local_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed_text(text: str) -> list[float]:
        return local_model.embed_query(text)

    def embed_documents(docs: list[str]) -> list[list[float]]:
        return local_model.embed_documents(docs)
