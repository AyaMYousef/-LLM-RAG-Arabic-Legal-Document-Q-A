"""Build BM25 sparse index from the processed article corpus.

The BM25 index is saved alongside the FAISS vector store and used by
HybridRetriever to improve recall on exact article-number queries and
rare legal terms.

Usage:
    python scripts/build_bm25_index.py
"""
from __future__ import annotations

import json
from pathlib import Path

from src.ingestion.bm25_store import BM25Store
from src.ingestion.chunker import build_chunk_text

CORPUS_PATH = Path("data/processed/corpus_raw.json")
BM25_DIR = Path("data/processed/bm25_store")


def main() -> None:
    print("=" * 80)
    print("BUILDING BM25 INDEX")
    print("=" * 80)

    print(f"\n[1/3] Loading corpus from {CORPUS_PATH}...")
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        articles = json.load(fh)
    print(f"Articles loaded: {len(articles)}")

    print("\n[2/3] Building corpus texts and metadata...")
    texts: list[str] = []
    metadata: list[dict] = []

    for article in articles:
        # Use the same bilingual chunk text as the dense index for consistency
        text = build_chunk_text(article)
        texts.append(text)
        metadata.append(
            {
                "article_number": article["article_number"],
                "citation": article["citation"],
                "source_page": article["source_page"],
                "book": article.get("book"),
                "chapter": article.get("chapter"),
                "section": article.get("section"),
                "topic_ar": article.get("topic_ar"),
                "topic_en": article.get("topic_en"),
                "is_repealed": article["is_repealed"],
                "text_en": article.get("text_en"),
                "text_ar": article.get("text_ar"),
            }
        )

    print(f"Texts prepared: {len(texts)}")

    print("\n[3/3] Building and saving BM25 index...")
    store = BM25Store(texts=texts, metadata=metadata)
    store.save(BM25_DIR)

    print(f"\nVectors indexed: {len(texts)}")
    print("\n" + "=" * 80)
    print("BM25 INDEX BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
