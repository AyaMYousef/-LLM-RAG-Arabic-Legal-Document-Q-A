from __future__ import annotations

import re
from pathlib import Path

import pymupdf


PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")

Y_TOLERANCE = 2.0


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def contains_latin(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text))


def extract_spans(page: pymupdf.Page) -> list[dict]:
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

    return spans


def group_spans_by_y(spans: list[dict]) -> list[list[dict]]:
    spans = sorted(
        spans,
        key=lambda span: (span["y0"], span["x0"]),
    )

    groups = []

    for span in spans:
        if not groups:
            groups.append([span])
            continue

        if abs(span["y0"] - groups[-1][0]["y0"]) <= Y_TOLERANCE:
            groups[-1].append(span)
        else:
            groups.append([span])

    return groups


def reconstruct_english_lines(spans: list[dict]) -> list[dict]:
    english_spans = [
        span
        for span in spans
        if contains_latin(span["text"])
        and not contains_arabic(span["text"])
    ]

    groups = group_spans_by_y(english_spans)

    lines = []

    for group in groups:
        ordered = sorted(
            group,
            key=lambda span: span["x0"],
        )

        text = normalize_spaces(
            " ".join(span["text"] for span in ordered)
        )

        lines.append(
            {
                "y": group[0]["y0"],
                "text": text,
            }
        )

    return lines


def detect_article(text: str) -> int | None:
    text = normalize_spaces(text)

    match = re.match(
        r"^Article\s+(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(1))


def main() -> None:
    doc = pymupdf.open(PDF_PATH)

    detections = []

    for page_number, page in enumerate(doc, start=1):
        spans = extract_spans(page)
        lines = reconstruct_english_lines(spans)

        for line in lines:
            article_number = detect_article(line["text"])

            if article_number is not None:
                detections.append(
                    {
                        "article": article_number,
                        "page": page_number,
                        "y": line["y"],
                        "text": line["text"],
                    }
                )

    for item in detections:
        print(
            f"Article {item['article']:4d} "
            f"| page {item['page']:3d} "
            f"| y={item['y']:7.2f} "
            f"| {item['text']}"
        )

    print()
    print("Total detections:", len(detections))
    print(
        "Unique article numbers:",
        len({item["article"] for item in detections}),
    )


if __name__ == "__main__":
    main()