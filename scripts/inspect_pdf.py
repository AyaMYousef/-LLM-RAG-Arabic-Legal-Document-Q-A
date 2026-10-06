from __future__ import annotations

import json
import re
from pathlib import Path

import fitz

PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("reports/pdf_inspection.json")


ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)


def normalize_article_number(value: str) -> int:
    """Convert Arabic-Indic or Western digits to an integer."""
    value = value.translate(ARABIC_DIGITS)
    value = re.sub(r"\D", "", value)

    if not value:
        raise ValueError(f"Invalid article number: {value!r}")

    return int(value)


def classify_block(x0: float, page_width: float) -> str:
    """
    Classify a PDF block according to its horizontal position.

    The current PDF is approximately:
        Arabic -> right column
        English -> left column
    """
    midpoint = page_width / 2

    if x0 >= midpoint:
        return "arabic"

    return "english"


def inspect_pdf() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Place the Egyptian Civil Code PDF in data/raw/"
        )

    doc = fitz.open(PDF_PATH)

    print(f"PDF: {PDF_PATH}")
    print(f"Pages: {len(doc)}")

    pages = []

    for page_number, page in enumerate(doc, start=1):
        page_width = page.rect.width
        page_height = page.rect.height

        blocks = []

        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text = block[:5]

            text = text.strip()

            if not text:
                continue

            column = classify_block(x0, page_width)

            blocks.append(
                {
                    "x0": round(x0, 2),
                    "y0": round(y0, 2),
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "column": column,
                    "text": text,
                }
            )

        blocks.sort(key=lambda item: (item["y0"], item["x0"]))

        pages.append(
            {
                "page": page_number,
                "width": page_width,
                "height": page_height,
                "blocks": blocks,
            }
        )

        if page_number <= 5:
            print(f"\n--- PAGE {page_number} ---")

            for block in blocks[:10]:
                print(
                    f"[{block['column']:7}] "
                    f"x={block['x0']:6.1f} "
                    f"y={block['y0']:6.1f} "
                    f"{block['text'][:100]!r}"
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

    print(f"\nInspection written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    inspect_pdf()