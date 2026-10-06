from __future__ import annotations

import re
from pathlib import Path

import pymupdf

PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)

Y_TOLERANCE = 2.0


def normalize_digits(text: str) -> str:
    return text.translate(ARABIC_DIGITS)


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


def classify_span(text: str) -> str:
    has_arabic = contains_arabic(text)
    has_latin = contains_latin(text)

    if has_arabic:
        return "arabic"

    if has_latin:
        return "english"

    return "other"


def group_spans_by_y(spans: list[dict]) -> list[list[dict]]:
    spans = sorted(
        spans,
        key=lambda span: (span["y0"], span["x0"]),
    )

    groups: list[list[dict]] = []

    for span in spans:
        if not groups:
            groups.append([span])
            continue

        current_y = groups[-1][0]["y0"]

        if abs(span["y0"] - current_y) <= Y_TOLERANCE:
            groups[-1].append(span)
        else:
            groups.append([span])

    return groups


def reconstruct_language_lines(
    spans: list[dict],
    language: str,
) -> list[dict]:
    """
    Reconstruct one language stream.

    Arabic:
        right -> left

    English:
        left -> right
    """

    language_spans = [
        span
        for span in spans
        if classify_span(span["text"]) == language
    ]

    groups = group_spans_by_y(language_spans)

    lines = []

    for group in groups:
        if language == "arabic":
            ordered = sorted(
                group,
                key=lambda span: span["x0"],
                reverse=True,
            )
        else:
            ordered = sorted(
                group,
                key=lambda span: span["x0"],
            )

        text = normalize_spaces(
            " ".join(
                span["text"]
                for span in ordered
            )
        )

        lines.append(
            {
                "y": round(group[0]["y0"], 2),
                "text": text,
            }
        )

    return lines


def detect_arabic_article(text: str) -> int | None:
    text = normalize_digits(text)
    text = normalize_spaces(text)

    cleaned = re.sub(
        r"[()\[\]{}]",
        " ",
        text,
    )

    cleaned = normalize_spaces(cleaned)

    match = re.match(
        r"^مادة\s+(\d+)\b",
        cleaned,
    )

    if match:
        return int(match.group(1))

    return None


def detect_english_article(text: str) -> int | None:
    text = normalize_spaces(text)

    match = re.match(
        r"^Article\s+(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    doc = pymupdf.open(PDF_PATH)

    arabic_articles = []
    english_articles = []

    for page_number, page in enumerate(doc, start=1):
        spans = extract_spans(page)

        arabic_lines = reconstruct_language_lines(
            spans,
            "arabic",
        )

        english_lines = reconstruct_language_lines(
            spans,
            "english",
        )

        for line_index, line in enumerate(arabic_lines):
            article_number = detect_arabic_article(
                line["text"]
            )

            if article_number is not None:
                arabic_articles.append(
                    {
                        "page": page_number,
                        "line_index": line_index,
                        "y": line["y"],
                        "article_number": article_number,
                        "text": line["text"],
                    }
                )

        for line_index, line in enumerate(english_lines):
            article_number = detect_english_article(
                line["text"]
            )

            if article_number is not None:
                english_articles.append(
                    {
                        "page": page_number,
                        "line_index": line_index,
                        "y": line["y"],
                        "article_number": article_number,
                        "text": line["text"],
                    }
                )

    print("\n=== ARABIC ARTICLE HEADERS ===")

    for article in arabic_articles:
        print(
            f"PAGE {article['page']:3} "
            f"Y={article['y']:7.2f} "
            f"Article {article['article_number']:4} "
            f"{article['text']!r}"
        )

    print("\n=== ENGLISH ARTICLE HEADERS ===")

    for article in english_articles:
        print(
            f"PAGE {article['page']:3} "
            f"Y={article['y']:7.2f} "
            f"Article {article['article_number']:4} "
            f"{article['text']!r}"
        )

    print()
    print(
        f"Arabic detections: {len(arabic_articles)}"
    )
    print(
        f"English detections: {len(english_articles)}"
    )


if __name__ == "__main__":
    main()