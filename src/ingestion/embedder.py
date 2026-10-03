from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class Embedder:
    """Generate embeddings for RAG documents."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Encode texts into normalized embedding vectors."""

        return self.model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

        return self.model.get_embedding_dimension()