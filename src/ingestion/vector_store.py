from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class FAISSVectorStore:
    """FAISS vector store with JSON metadata."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        metadata: list[dict[str, Any]],
    ) -> None:
        """Add normalized embeddings and corresponding metadata."""

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}."
            )

        if len(embeddings) != len(metadata):
            raise ValueError(
                "Number of embeddings must match number of metadata records."
            )

        self.index.add(
            np.asarray(embeddings, dtype=np.float32)
        )
        self.metadata.extend(metadata)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k matching metadata records."""

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[index],
                }
            )

        return results

    def save(self, directory: str | Path) -> None:
        """Save FAISS index and metadata to disk."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        with open(
            directory / "metadata.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path) -> "FAISSVectorStore":
        """Load a previously saved vector store."""

        directory = Path(directory)

        index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata

        return store