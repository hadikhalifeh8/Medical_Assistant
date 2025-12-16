import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)

def embed_text(text: str) -> list[float]:
    response = client.feature_extraction(
        text,
        model="intfloat/e5-small-v2"
    )

    # HF may return nested list
    if isinstance(response[0], list):
        return response[0]

    return response
