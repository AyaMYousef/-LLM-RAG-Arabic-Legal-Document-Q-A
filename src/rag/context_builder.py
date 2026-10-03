from __future__ import annotations

from typing import Any


def build_context(results: list[dict[str, Any]]) -> str:
    """Build LLM-ready context from retrieved legal articles."""

    parts = []

    for result in results:
        metadata = result["metadata"]

        citation = metadata["citation"]
        source_page = metadata.get("source_page")

        text_en = metadata.get("text_en") or ""
        text_ar = metadata.get("text_ar") or ""

        part = (
            f"{citation}\n"
            f"PDF page: {source_page}\n\n"
            f"English:\n{text_en}\n\n"
            f"العربية:\n{text_ar}"
        )

        parts.append(part)

    return "\n\n---\n\n".join(parts)