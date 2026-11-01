"""Build the multilingual E5 FAISS vector index.

Embeddings are cached to disk so that re-runs after interruptions skip
the expensive encoding step when the corpus has not changed.

Cache files:
    data/processed/embeddings_cache.npy   – raw embedding matrix
    data/processed/corpus_hash.txt        – sha256 of corpus_raw.json

If the corpus hash matches the stored hash, cached embeddings are loaded
instead of re-encoding.  Delete these files to force a full rebuild.

Usage:
    python scripts/build_e5_index.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import build_chunks
from src.ingestion.vector_store import FAISSVectorStore

CORPUS_PATH = Path("data/processed/corpus_raw.json")
INDEX_DIR = Path("data/processed/vector_store_e5")
MODEL_NAME = "intfloat/multilingual-e5-base"

CACHE_EMBEDDINGS = Path("data/processed/embeddings_cache.npy")
CACHE_HASH = Path("data/processed/corpus_hash.txt")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _corpus_sha256(corpus_path: Path) -> str:
    """Return the sha256 hex digest of the raw corpus file."""
    h = hashlib.sha256()
    with corpus_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cached_embeddings(
    corpus_hash: str,
) -> np.ndarray | None:
    """Return cached embeddings if the corpus hash matches, else None."""
    if not CACHE_EMBEDDINGS.exists() or not CACHE_HASH.exists():
        return None

    stored_hash = CACHE_HASH.read_text().strip()
    if stored_hash != corpus_hash:
        print(
            f"  Corpus changed (hash mismatch).  "
            f"Stored: {stored_hash[:12]}…  Current: {corpus_hash[:12]}…"
        )
        return None

    embeddings = np.load(CACHE_EMBEDDINGS)
    print(
        f"  Cache hit — loaded {embeddings.shape[0]} embeddings "
        f"from {CACHE_EMBEDDINGS}"
    )
    return embeddings


def _save_cached_embeddings(
    embeddings: np.ndarray,
    corpus_hash: str,
) -> None:
    """Persist embeddings and corpus hash for future runs."""
    CACHE_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_EMBEDDINGS, embeddings)
    CACHE_HASH.write_text(corpus_hash)
    print(f"  Embeddings cached → {CACHE_EMBEDDINGS}")
    print(f"  Corpus hash stored → {CACHE_HASH}")


# ---------------------------------------------------------------------------
# Main build routine
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("BUILDING E5 VECTOR INDEX")
    print("=" * 80)

    # [1] Load corpus
    print(f"\n[1/5] Loading corpus from {CORPUS_PATH}...")
    with CORPUS_PATH.open(encoding="utf-8") as file:
        articles = json.load(file)
    print(f"Articles loaded: {len(articles)}")

    # [2] Build chunks
    print("\n[2/5] Building article-level chunks...")
    chunks = build_chunks(articles)
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No chunks were created.")

    if len(chunks) != len(articles):
        raise ValueError(
            "Number of chunks does not match number of articles."
        )

    # [3] Load E5 model
    print(f"\n[3/5] Loading E5 model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)
    dimension = model.get_embedding_dimension()
    print(f"Dimension: {dimension}")

    # [4] Generate or load cached embeddings
    print("\n[4/5] Generating E5 passage embeddings...")

    corpus_hash = _corpus_sha256(CORPUS_PATH)
    print(f"  Corpus sha256: {corpus_hash[:16]}…")

    embeddings = _load_cached_embeddings(corpus_hash)

    if embeddings is None:
        print("  No valid cache found — encoding corpus (this may take a few minutes)...")
        texts = [
            f"passage: {chunk['text']}"
            for chunk in chunks
        ]

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        _save_cached_embeddings(embeddings, corpus_hash)
    else:
        print("  Skipped encoding — using cached embeddings.")

    print(f"  Embedding shape: {embeddings.shape}")

    expected_shape = (len(chunks), dimension)
    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape. "
            f"Expected {expected_shape}, got {embeddings.shape}."
        )

    # [5] Build and save FAISS index
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
    store.add(embeddings=embeddings, metadata=metadata)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(INDEX_DIR)

    print(f"\nIndex saved → {INDEX_DIR}")
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
