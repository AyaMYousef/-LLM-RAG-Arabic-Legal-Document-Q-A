"""Retriever classes for the Arabic Legal RAG pipeline.

Two retrievers are available:

- ``Retriever``       – dense-only (multilingual E5 + FAISS).
- ``HybridRetriever`` – dense + BM25 sparse fusion for improved recall on
                        exact article-number lookups and rare legal terms.

The existing ``Retriever`` is unchanged so all call-sites continue to work.
``HybridRetriever`` wraps it and adds a ``BM25Store`` loaded from disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import FAISSVectorStore

DEFAULT_E5_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_VECTOR_STORE = "data/processed/vector_store_e5"
DEFAULT_BM25_STORE = "data/processed/bm25_store"


# ---------------------------------------------------------------------------
# Dense-only Retriever (original – unchanged)
# ---------------------------------------------------------------------------

class Retriever:
    """Dense retriever using multilingual E5 embeddings and FAISS."""

    def __init__(
        self,
        vector_store_path: str = DEFAULT_VECTOR_STORE,
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
        print("EMBEDDING MODEL:", self.embedder.model_name)

        self.vector_store = FAISSVectorStore.load(resolved_path)

    def search(
        self,
        question: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        query = f"query: {question}"

        print("=== SEARCH DEBUG ===")
        print("Question:", question)
        print("Query:", query)
        print("Model:", self.embedder.model_name)

        embedding = self.embedder.encode([query])[0]

        print("Embedding dimension:", len(embedding))
        print("First 5 values:", embedding[:5])
        print("====================")

        return self.vector_store.search(
            embedding,
            k=k,
        )


# ---------------------------------------------------------------------------
# Hybrid Retriever (dense + BM25 fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retriever that fuses E5 dense scores with BM25 sparse scores.

    Use this when you need improved recall on:
    - Exact article-number lookup queries  (e.g. "المادة 1030")
    - Rare legal terms with low embedding coverage

    Parameters
    ----------
    vector_store_path:
        Directory containing the FAISS index and metadata JSON.
    bm25_store_path:
        Directory containing the BM25 index (built by build_bm25_index.py).
    model_name:
        Sentence-transformer model name for dense encoding.
    alpha:
        Fusion weight: ``alpha=1.0`` is dense-only, ``alpha=0.0`` is BM25-only.
        Default 0.6 puts more weight on semantic similarity while still
        benefiting from exact-term matching.
    fetch_k:
        Candidate pool drawn from each index before score fusion.
    """

    def __init__(
        self,
        vector_store_path: str = DEFAULT_VECTOR_STORE,
        bm25_store_path: str = DEFAULT_BM25_STORE,
        model_name: str = DEFAULT_E5_MODEL,
        alpha: float = 0.6,
        fetch_k: int = 20,
    ) -> None:
        from src.ingestion.bm25_store import BM25Store

        print("=== HYBRID RETRIEVER ===")
        print("alpha (dense weight):", alpha)
        print("fetch_k:", fetch_k)

        self.alpha = alpha
        self.fetch_k = fetch_k

        self.embedder = Embedder(model_name=model_name)
        print("Embedding model:", self.embedder.model_name)

        v_path = Path(vector_store_path).resolve()
        print("Vector store:", v_path)
        self.vector_store = FAISSVectorStore.load(v_path)

        b_path = Path(bm25_store_path).resolve()
        print("BM25 store:", b_path)
        self.bm25_store = BM25Store.load(b_path)

        print("========================")

    def search(
        self,
        question: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return top-k results fusing dense and BM25 scores."""
        if not question.strip():
            raise ValueError("Question must not be empty.")

        query = f"query: {question}"
        embedding = self.embedder.encode([query])[0]

        return self.bm25_store.hybrid_search(
            dense_store=self.vector_store,
            query_embedding=embedding,
            query_text=question,
            k=k,
            alpha=self.alpha,
            fetch_k=self.fetch_k,
        )
