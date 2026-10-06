from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.chunker import build_chunks
from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import FAISSVectorStore

CORPUS_PATH = Path("data/processed/corpus_raw.json")
INDEX_DIR = Path("data/processed/vector_store")


def main() -> None:
    print("=" * 80)
    print("BUILDING VECTOR INDEX")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load corpus
    # ------------------------------------------------------------------
    print("\n[1/5] Loading corpus...")

    with CORPUS_PATH.open(encoding="utf-8") as file:
        articles = json.load(file)

    print(f"Articles loaded: {len(articles)}")

    # ------------------------------------------------------------------
    # 2. Build article-level chunks
    # ------------------------------------------------------------------
    print("\n[2/5] Building article-level chunks...")

    chunks = build_chunks(articles)

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created.")

    if len(chunks) != len(articles):
        raise ValueError(
            "Number of chunks does not match number of articles."
        )

    # ------------------------------------------------------------------
    # 3. Load embedding model
    # ------------------------------------------------------------------
    print("\n[3/5] Loading embedding model...")

    embedder = Embedder()

    print(f"Model: {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")

    # ------------------------------------------------------------------
    # 4. Generate embeddings
    # ------------------------------------------------------------------
    print("\n[4/5] Generating embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.encode(texts)

    print(f"Embedding shape: {embeddings.shape}")

    expected_shape = (len(chunks), embedder.dimension)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape. "
            f"Expected {expected_shape}, got {embeddings.shape}."
        )

    # ------------------------------------------------------------------
    # 5. Build and save FAISS index
    # ------------------------------------------------------------------
    print("\n[5/5] Building FAISS index...")

    metadata = [
    {
        **chunk["metadata"],
        "text_en": article.get("text_en"),
        "text_ar": article.get("text_ar"),
    }
    for article, chunk in zip(articles, chunks)
]
    store = FAISSVectorStore(
        dimension=embedder.dimension,
    )

    store.add(
        embeddings=embeddings,
        metadata=metadata,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    store.save(INDEX_DIR)

    print(f"Index saved to: {INDEX_DIR}")
    print(f"Vectors indexed: {store.index.ntotal}")
    print(f"Metadata records: {len(store.metadata)}")

    # ------------------------------------------------------------------
    # Final verification
    # ------------------------------------------------------------------
    if store.index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index size does not match number of chunks."
        )

    print("\n" + "=" * 80)
    print("INDEX BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()