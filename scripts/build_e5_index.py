from __future__ import annotations

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import build_chunks
from src.ingestion.vector_store import FAISSVectorStore


CORPUS_PATH = Path("data/processed/corpus_raw.json")
INDEX_DIR = Path("data/processed/vector_store_e5")
MODEL_NAME = "intfloat/multilingual-e5-base"


def main() -> None:
    print("=" * 80)
    print("BUILDING E5 VECTOR INDEX")
    print("=" * 80)

    print("\n[1/5] Loading corpus...")

    with CORPUS_PATH.open(encoding="utf-8") as file:
        articles = json.load(file)

    print(f"Articles loaded: {len(articles)}")

    print("\n[2/5] Building article-level chunks...")

    chunks = build_chunks(articles)

    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created.")

    if len(chunks) != len(articles):
        raise ValueError(
            "Number of chunks does not match number of articles."
        )

    print("\n[3/5] Loading E5 model...")

    model = SentenceTransformer(MODEL_NAME)

    dimension = model.get_embedding_dimension()

    print(f"Model: {MODEL_NAME}")
    print(f"Dimension: {dimension}")

    print("\n[4/5] Generating E5 passage embeddings...")

    texts = [
        f"passage: {chunk['text']}"
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Embedding shape: {embeddings.shape}")

    expected_shape = (len(chunks), dimension)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape. "
            f"Expected {expected_shape}, "
            f"got {embeddings.shape}."
        )

    print("\n[5/5] Building FAISS index...")

    metadata = [
    {
        **chunk["metadata"],
        "text_en": article.get("text_en"),
        "text_ar": article.get("text_ar"),
    }
    for article, chunk in zip(articles, chunks)
]

    store = FAISSVectorStore(dimension=dimension)

    store.add(
        embeddings=embeddings,
        metadata=metadata,
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    store.save(INDEX_DIR)

    print(f"Index saved to: {INDEX_DIR}")
    print(f"Vectors indexed: {store.index.ntotal}")
    print(f"Metadata records: {len(store.metadata)}")

    if store.index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index size does not match number of chunks."
        )

    print("\n" + "=" * 80)
    print("E5 INDEX BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()