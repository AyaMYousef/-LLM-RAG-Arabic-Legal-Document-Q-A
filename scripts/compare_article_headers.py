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
    if contains_arabic(text):
        return "arabic"

    if contains_latin(text):
        return "english"

    return "other"


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


def reconstruct_language_lines(
    spans: list[dict],
    language: str,
) -> list[dict]:
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
                "y": group[0]["y0"],
                "text": text,
            }
        )

    return lines


def detect_arabic_article(text: str) -> int | None:
    text = normalize_digits(text)
    text = normalize_spaces(text)

    text = re.sub(
        r"[()\[\]{}]",
        " ",
        text,
    )

    text = normalize_spaces(text)

    match = re.match(
        r"^مادة\s+(\d+)\b",
        text,
    )

    return int(match.group(1)) if match else None


def detect_english_article(text: str) -> int | None:
    text = normalize_spaces(text)

    match = re.match(
        r"^Article\s+(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )

    return int(match.group(1)) if match else None


def main() -> None:
    doc = pymupdf.open(PDF_PATH)

    arabic: dict[int, list[dict]] = {}
    english: dict[int, list[dict]] = {}

    for page_number, page in enumerate(doc, start=1):
        spans = extract_spans(page)

        for language, target, detector in [
            (
                "arabic",
                arabic,
                detect_arabic_article,
            ),
            (
                "english",
                english,
                detect_english_article,
            ),
        ]:
            lines = reconstruct_language_lines(
                spans,
                language,
            )

            for line in lines:
                number = detector(line["text"])

                if number is not None:
                    target.setdefault(number, []).append(
                        {
                            "page": page_number,
                            "y": line["y"],
                            "text": line["text"],
                        }
                    )

    arabic_numbers = set(arabic)
    english_numbers = set(english)

    print("Arabic count:", len(arabic_numbers))
    print("English count:", len(english_numbers))

    print("\nMissing from Arabic:")
    for number in sorted(english_numbers - arabic_numbers):
        print(
            f"Article {number}: "
            f"English={english[number]}"
        )

    print("\nMissing from English:")
    for number in sorted(arabic_numbers - english_numbers):
        print(
            f"Article {number}: "
            f"Arabic={arabic[number]}"
        )

    print("\nDuplicate Arabic headers:")
    for number, occurrences in sorted(arabic.items()):
        if len(occurrences) > 1:
            print(number, occurrences)

    print("\nDuplicate English headers:")
    for number, occurrences in sorted(english.items()):
        if len(occurrences) > 1:
            print(number, occurrences)


if __name__ == "__main__":
    main()