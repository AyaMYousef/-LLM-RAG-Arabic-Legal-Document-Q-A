from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import FAISSVectorStore

DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"

def __init__(
    self,
    vector_store_path: str = "data/processed/vector_store_e5",
    model_name: str = DEFAULT_E5_MODEL,
) -> None:
    print("=== RETRIEVER DEBUG ===")
    print("Current working directory:", Path.cwd())

    resolved_path = Path(vector_store_path).resolve()

    print("Vector store input:", vector_store_path)
    print("Vector store resolved:", resolved_path)
    print("Vector store exists:", resolved_path.exists())
    print("=======================")

    self.embedder = Embedder(model_name=model_name)
    self.vector_store = FAISSVectorStore.load(resolved_path)