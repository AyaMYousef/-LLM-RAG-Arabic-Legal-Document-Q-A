from __future__ import annotations

import json
import re
from pathlib import Path

import fitz


PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("reports/column_text.json")


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def is_arabic(text: str) -> bool:
    """Return True when the text contains Arabic characters."""
    return bool(ARABIC_RE.search(text))


def is_english(text: str) -> bool:
    """Return True when the text contains Latin characters."""
    return bool(re.search(r"[A-Za-z]", text))


def classify_span(text: str) -> str:
    """
    Classify a span based on the characters it contains.

    This is more reliable than classifying by x-coordinate because
    the PDF sometimes places Arabic and English text in the same
    visual block or line.
    """
    has_arabic = is_arabic(text)
    has_english = is_english(text)

    if has_arabic and has_english:
        return "mixed"

    if has_arabic:
        return "arabic"

    if has_english:
        return "english"

    return "other"


def extract_page(page: fitz.Page) -> dict:
    page_dict = page.get_text("dict")

    spans = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()

                if not text:
                    continue

                x0, y0, x1, y1 = span["bbox"]

                spans.append(
                    {
                        "x0": round(x0, 2),
                        "y0": round(y0, 2),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "text": text,
                        "language": classify_span(text),
                    }
                )

    return {
        "page": page.number + 1,
        "spans": spans,
    }


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)

    pages = []

    for page in doc:
        page_data = extract_page(page)
        pages.append(page_data)

        if page.number < 5:
            print(f"\n--- PAGE {page.number + 1} ---")

            for span in page_data["spans"]:
                print(
                    f"[{span['language']:7}] "
                    f"x={span['x0']:7.2f} "
                    f"y={span['y0']:7.2f} "
                    f"{span['text']}"
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pdf": str(PDF_PATH),
                "pages": len(doc),
                "pages_data": pages,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nWritten to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()