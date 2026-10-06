from __future__ import annotations

import json
from pathlib import Path

import fitz

PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("reports/pdf_spans.json")


def inspect_pdf() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        page_data = {
            "page": page_number,
            "width": page.rect.width,
            "height": page.rect.height,
            "spans": [],
        }

        page_dict = page.get_text("dict")

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()

                    if not text:
                        continue

                    bbox = span["bbox"]

                    page_data["spans"].append(
                        {
                            "x0": round(bbox[0], 2),
                            "y0": round(bbox[1], 2),
                            "x1": round(bbox[2], 2),
                            "y1": round(bbox[3], 2),
                            "text": text,
                            "font": span.get("font"),
                            "size": round(span.get("size", 0), 2),
                        }
                    )

        pages.append(page_data)

        if page_number <= 5:
            print(f"\n--- PAGE {page_number} ---")

            for span in page_data["spans"]:
                print(
                    f"x={span['x0']:7.2f} "
                    f"y={span['y0']:7.2f} "
                    f"size={span['size']:5.1f} "
                    f"{span['text']!r}"
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

    print(f"\nSpan inspection written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    inspect_pdf()