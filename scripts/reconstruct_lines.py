from __future__ import annotations

import json
from pathlib import Path

import fitz

PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("reports/reconstructed_lines.json")

Y_TOLERANCE = 2.0


def reconstruct_page_lines(page: fitz.Page) -> list[dict]:
    """
    Reconstruct visual lines from individual PDF spans.

    Arabic lines are read from right to left.
    English lines are read from left to right.

    We first group spans by approximately equal Y coordinates,
    then order spans horizontally according to their column.
    """
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
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "text": text,
                    }
                )

    # Sort primarily by vertical position.
    spans.sort(key=lambda span: (span["y0"], span["x0"]))

    lines: list[dict] = []

    for span in spans:
        matching_line = None

        for line in lines:
            if abs(line["y"] - span["y0"]) <= Y_TOLERANCE:
                matching_line = line
                break

        if matching_line is None:
            matching_line = {
                "y": span["y0"],
                "spans": [],
            }
            lines.append(matching_line)

        matching_line["spans"].append(span)

    reconstructed = []

    for line in lines:
        line_spans = line["spans"]

        # Determine whether this is primarily Arabic.
        # The Arabic column is generally on the right side.
        average_x = sum(span["x0"] for span in line_spans) / len(line_spans)

        if average_x > page.rect.width / 2:
            direction = "rtl"
            ordered = sorted(
                line_spans,
                key=lambda span: span["x0"],
                reverse=True,
            )
        else:
            direction = "ltr"
            ordered = sorted(
                line_spans,
                key=lambda span: span["x0"],
            )

        text = " ".join(span["text"] for span in ordered)

        reconstructed.append(
            {
                "y": round(line["y"], 2),
                "direction": direction,
                "x_min": round(min(span["x0"] for span in line_spans), 2),
                "x_max": round(max(span["x1"] for span in line_spans), 2),
                "text": text,
            }
        )

    reconstructed.sort(key=lambda line: line["y"])

    return reconstructed


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        lines = reconstruct_page_lines(page)

        pages.append(
            {
                "page": page_number,
                "lines": lines,
            }
        )

        if page_number <= 5:
            print(f"\n--- PAGE {page_number} ---")

            for line in lines:
                print(
                    f"[{line['direction']}] "
                    f"y={line['y']:7.2f} "
                    f"{line['text']}"
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