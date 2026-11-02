"""BM25 sparse index for hybrid dense+sparse retrieval.

Builds and persists a BM25 index over the article corpus.  During
hybrid search the BM25 term-match scores are fused with FAISS cosine
similarity scores using a configurable alpha weight:

    final_score = alpha * dense_score + (1 - alpha) * bm25_score_normalised

BM25 improves retrieval for exact article-number lookups and rare
legal terms that semantic embeddings tend to miss.

Build the index:
    python scripts/build_bm25_index.py

Then evaluate:
    python scripts/evaluate_retrieval.py --retriever hybrid
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.vector_store import FAISSVectorStore

# ---------------------------------------------------------------------------
# Simple bilingual tokeniser
# ---------------------------------------------------------------------------

_AR_WORD = re.compile(r"[\u0600-\u06FF\u0750-\u077F]+")
_EN_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenise(text: str) -> list[str]:
    """Tokenise Arabic and/or English text into lowercase word tokens."""
    tokens = _AR_WORD.findall(text) + [t.lower() for t in _EN_WORD.findall(text)]
    return tokens


# ---------------------------------------------------------------------------
# BM25Store
# ---------------------------------------------------------------------------

class BM25Store:
    """BM25 index aligned with a FAISSVectorStore metadata list."""

    _INDEX_FILE = "bm25_index.pkl"
    _CORPUS_FILE = "bm25_corpus.json"

    def __init__(
        self,
        texts: list[str],
        metadata: list[dict[str, Any]],
    ) -> None:
        self.metadata = metadata
        self._texts = texts
        tokenised = [tokenise(t) for t in texts]
        self.bm25 = BM25Okapi(tokenised)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Return top-k results by BM25 score only."""
        tokens = tokenise(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            {"score": float(scores[i]), "metadata": self.metadata[i]}
            for i in top_indices
        ]

    def hybrid_search(
        self,
        dense_store: FAISSVectorStore,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 5,
        alpha: float = 0.6,
        fetch_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Fuse FAISS dense scores with BM25 sparse scores.

        Parameters
        ----------
        dense_store:
            Loaded FAISSVectorStore for cosine-similarity search.
        query_embedding:
            Normalised query vector (shape ``(1, dim)`` or ``(dim,)``).
        query_text:
            Raw query string used for BM25 scoring.
        k:
            Number of results to return.
        alpha:
            Weight on dense score (0 = BM25 only, 1 = dense only).
        fetch_k:
            Candidate pool size from each index before fusion.
        """
        # --- Dense candidates ---
        dense_results = dense_store.search(query_embedding, k=fetch_k)
        dense_score_map: dict[int, float] = {}
        for r in dense_results:
            art_num = r["metadata"]["article_number"]
            dense_score_map[art_num] = r["score"]

        # Normalise dense scores to [0, 1]
        if dense_score_map:
            d_min = min(dense_score_map.values())
            d_max = max(dense_score_map.values())
            d_range = d_max - d_min or 1.0
            dense_score_map = {
                a: (s - d_min) / d_range
                for a, s in dense_score_map.items()
            }

        # --- BM25 candidates ---
        bm25_results = self.search(query_text, k=fetch_k)
        bm25_score_map: dict[int, float] = {}
        for r in bm25_results:
            art_num = r["metadata"]["article_number"]
            bm25_score_map[art_num] = r["score"]

        # Normalise BM25 scores to [0, 1]
        if bm25_score_map:
            b_min = min(bm25_score_map.values())
            b_max = max(bm25_score_map.values())
            b_range = b_max - b_min or 1.0
            bm25_score_map = {
                a: (s - b_min) / b_range
                for a, s in bm25_score_map.items()
            }

        # --- Fuse scores ---
        all_articles = set(dense_score_map) | set(bm25_score_map)
        fused: dict[int, float] = {
            art: (
                alpha * dense_score_map.get(art, 0.0)
                + (1 - alpha) * bm25_score_map.get(art, 0.0)
            )
            for art in all_articles
        }

        top_articles = sorted(fused, key=fused.__getitem__, reverse=True)[:k]

        # Re-attach metadata from BM25 store (aligned by article_number)
        art_to_meta: dict[int, dict[str, Any]] = {
            m["article_number"]: m for m in self.metadata
        }

        return [
            {
                "score": fused[art],
                "dense_score": dense_score_map.get(art, 0.0),
                "bm25_score": bm25_score_map.get(art, 0.0),
                "metadata": art_to_meta[art],
            }
            for art in top_articles
            if art in art_to_meta
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist BM25 index and corpus text to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / self._INDEX_FILE, "wb") as fh:
            pickle.dump(self.bm25, fh)

        with open(directory / self._CORPUS_FILE, "w", encoding="utf-8") as fh:
            json.dump(
                {"texts": self._texts, "metadata": self.metadata},
                fh,
                ensure_ascii=False,
                indent=2,
            )

        print(f"BM25 index saved → {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> BM25Store:
        """Load a previously saved BM25Store from *directory*."""
        directory = Path(directory)

        with open(directory / cls._INDEX_FILE, "rb") as fh:
            bm25_obj = pickle.load(fh)

        with open(directory / cls._CORPUS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)

        store = cls.__new__(cls)
        store._texts = data["texts"]
        store.metadata = data["metadata"]
        store.bm25 = bm25_obj
        return store
