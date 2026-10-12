"""
Batch re-indexing script for the Arabic Legal RAG pipeline.

Adds one or more new JSON documents to the existing FAISS vector store
without rebuilding the entire index from scratch.

Each new document must follow the same schema as corpus_raw.json:
    article_number  (int)
    citation        (str)
    source_page     (int)
    is_repealed     (bool)
    text_ar         (str, optional)
    text_en         (str, optional)
    book            (str, optional)
    chapter         (str, optional)
    section         (str, optional)
    topic_ar        (str, optional)
    topic_en        (str, optional)

Usage - add articles from a JSON file:
    python scripts/batch_reindex.py --input data/raw/new_articles.json

Usage - add a single inline demo article (for quick tests):
    python scripts/batch_reindex.py --demo

The script:
    1. Loads the existing FAISS index from data/processed/vector_store_e5/
    2. Embeds the new articles using intfloat/multilingual-e5-base
    3. Appends the new vectors and metadata to the index
    4. Saves the updated index back to the same directory
    5. Prints a short summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── project root on sys.path so src imports work ───────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.chunker import build_chunks
from src.ingestion.vector_store import FAISSVectorStore

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "processed" / "vector_store_e5"

# ── demo article used when --demo is passed ────────────────────────────────
DEMO_ARTICLE = {
    "article_number": 9999,
    "citation": "Egyptian Civil Code, Article 9999 (Demo)",
    "source_page": 1,
    "is_repealed": False,
    "text_ar": "هذه مادة تجريبية تُستخدم لاختبار إعادة الفهرسة.",
    "text_en": "This is a demo article used to test batch re-indexing.",
    "book": None,
    "chapter": None,
    "section": "Demo",
    "topic_ar": "تجريبي",
    "topic_en": "Demo",
}


def load_articles(path: Path) -> list[dict]:
    """Load articles from a JSON file (list or single object)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return data


def embed_chunks(chunks: list[dict]):
    """Embed chunk texts using the E5 model with passage: prefix."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add new articles to the existing FAISS vector store."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a JSON file containing one or more article records.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Add a single demo article to verify the pipeline.",
    )
    args = parser.parse_args()

    if not args.demo and args.input is None:
        parser.print_help()
        sys.exit(1)

    # ── 1. load new articles ───────────────────────────────────────────────
    if args.demo:
        articles = [DEMO_ARTICLE]
        print("Demo mode: adding 1 demo article.")
    else:
        articles = load_articles(args.input)
        print(f"Loaded {len(articles)} article(s) from {args.input}")

    # ── 2. build chunks ────────────────────────────────────────────────────
    chunks = build_chunks(articles)
    print(f"Built {len(chunks)} chunk(s).")

    # ── 3. embed ───────────────────────────────────────────────────────────
    print(f"Embedding with {EMBEDDING_MODEL} ...")
    embeddings = embed_chunks(chunks)

    # ── 4. load existing index ─────────────────────────────────────────────
    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        print(
            f"ERROR: No existing index found at {VECTOR_STORE_DIR}.\n"
            "Run scripts/build_e5_index.py first."
        )
        sys.exit(1)

    print(f"Loading existing index from {VECTOR_STORE_DIR} ...")
    store = FAISSVectorStore.load(VECTOR_STORE_DIR)
    before = store.index.ntotal
    print(f"Index currently contains {before} vectors.")

    # ── 5. append new vectors and metadata ────────────────────────────────
    metadata = [chunk["metadata"] for chunk in chunks]

    # Attach full text to metadata so it can be used in context building
    for chunk, meta in zip(chunks, metadata):
        meta["text_en"] = next(
            (
                line.replace("English:\n", "")
                for line in chunk["text"].split("\n\n")
                if line.startswith("English:")
            ),
            "",
        )
        meta["text_ar"] = next(
            (
                line.replace("العربية:\n", "")
                for line in chunk["text"].split("\n\n")
                if line.startswith("العربية:")
            ),
            "",
        )

    store.add(embeddings, metadata)
    after = store.index.ntotal
    print(f"Added {after - before} vector(s). Index now has {after} vectors.")

    # ── 6. save ────────────────────────────────────────────────────────────
    store.save(VECTOR_STORE_DIR)
    print(f"Index saved to {VECTOR_STORE_DIR}")

    # ── 7. summary ─────────────────────────────────────────────────────────
    print("\n── Batch re-index complete ──")
    print(f"  Articles added : {len(articles)}")
    print(f"  Vectors before : {before}")
    print(f"  Vectors after  : {after}")
    print(f"  Index path     : {VECTOR_STORE_DIR / 'index.faiss'}")


if __name__ == "__main__":
    main()
