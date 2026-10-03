from __future__ import annotations

from typing import Any

from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import FAISSVectorStore


DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"


class Retriever:
    def __init__(
        self,
        vector_store_path: str = "data/processed/vector_store_e5",
        model_name: str = DEFAULT_E5_MODEL,
    ) -> None:
        self.embedder = Embedder(model_name=model_name)
        self.vector_store = FAISSVectorStore.load(vector_store_path)

    def search(
        self,
        question: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        query = f"query: {question}"

        embedding = self.embedder.encode([query])[0]

        return self.vector_store.search(
            embedding,
            k=k,
        )