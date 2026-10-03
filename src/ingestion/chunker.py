from __future__ import annotations

from typing import Any


def build_chunk_text(article: dict[str, Any]) -> str:
    """Build the bilingual text used as the embedding input."""

    parts = [
        article["citation"],
    ]

    if article.get("topic_en"):
        parts.append(f"Topic: {article['topic_en']}")

    if article.get("topic_ar"):
        parts.append(f"الموضوع: {article['topic_ar']}")

    if article.get("text_en"):
        parts.append(f"English:\n{article['text_en']}")

    if article.get("text_ar"):
        parts.append(f"العربية:\n{article['text_ar']}")

    return "\n\n".join(parts)


def build_chunk(article: dict[str, Any]) -> dict[str, Any]:
    """Convert one corpus article into an embedding-ready chunk."""

    metadata = {
        "article_number": article["article_number"],
        "citation": article["citation"],
        "source_page": article["source_page"],
        "book": article.get("book"),
        "chapter": article.get("chapter"),
        "section": article.get("section"),
        "topic_ar": article.get("topic_ar"),
        "topic_en": article.get("topic_en"),
        "is_repealed": article["is_repealed"],
    }

    return {
        "text": build_chunk_text(article),
        "metadata": metadata,
    }


def build_chunks(
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert the complete article-level corpus into chunks."""

    return [build_chunk(article) for article in articles]