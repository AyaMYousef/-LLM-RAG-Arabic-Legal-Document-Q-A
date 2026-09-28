from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf


PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("data/processed/corpus_raw.json")

Y_TOLERANCE = 2.0

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)

# Articles explicitly identified as repealed in the PDF.
#
# 55–80:
#   The PDF states that Articles 55–80 are repealed.
#
# 389–417:
#   The PDF states "Articles 389-417 repealed".
REPEALED_RANGES = (
    range(55, 81),
    range(389, 418),
)


def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits to Western digits."""
    return text.translate(ARABIC_DIGITS)


def normalize_spaces(text: str) -> str:
    """Collapse consecutive whitespace and trim the result."""
    return re.sub(r"\s+", " ", text).strip()


def contains_arabic(text: str) -> bool:
    """Return True if text contains Arabic Unicode characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def contains_latin(text: str) -> bool:
    """Return True if text contains Latin alphabet characters."""
    return bool(re.search(r"[A-Za-z]", text))


def is_repealed(article_number: int) -> bool:
    """Return True if the article belongs to a known repealed range."""
    return any(
        article_number in repealed_range
        for repealed_range in REPEALED_RANGES
    )


def extract_spans(page: pymupdf.Page) -> list[dict]:
    """Extract non-empty text spans with their bounding boxes."""
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
    """
    Group spans that belong to the same visual line.

    Spans are first sorted by vertical position and then
    horizontal position.
    """
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
    """
    Reconstruct English lines from English-only spans.

    English text is read from left to right.
    """
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


def reconstruct_arabic_lines(spans: list[dict]) -> list[dict]:
    """
    Reconstruct Arabic lines from Arabic-only spans.

    Arabic text is stored/extracted in RTL visual order,
    so spans are ordered from right to left.
    """
    arabic_spans = [
        span
        for span in spans
        if contains_arabic(span["text"])
        and not contains_latin(span["text"])
    ]

    groups = group_spans_by_y(arabic_spans)

    lines = []

    for group in groups:
        ordered = sorted(
            group,
            key=lambda span: span["x0"],
            reverse=True,
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


def detect_english_article(text: str) -> int | None:
    """
    Detect an English article header.

    Normally the PDF contains:
        Article 452

    In some locations PyMuPDF loses the first character:
        rticle 452
    """
    text = normalize_spaces(text)

    match = re.match(
        r"^(?:Article|rticle)\s*(\d+)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return int(match.group(1))


def detect_arabic_article(text: str) -> int | None:
    text = normalize_spaces(text)

    text = re.sub(
        r"[()\[\]{}]",
        " ",
        text,
    )

    text = normalize_spaces(text)

    # The PDF sometimes extracts "مادة" as "ما دة".
    match = re.match(
        r"^(?:مادة|ما\s+دة)\s+([٠-٩0-9]+)\b",
        text,
    )

    if not match:
        return None

    digits = normalize_digits(match.group(1))
    digits = digits[::-1]

    return int(digits)



def reconstruct_language_lines(
    spans: list[dict],
    language: str,
) -> list[dict]:
    """Reconstruct lines for one language."""

    if language == "en":
        selected = [
            span
            for span in spans
            if contains_latin(span["text"])
            and not contains_arabic(span["text"])
        ]
    elif language == "ar":
        selected = [
            span
            for span in spans
            if contains_arabic(span["text"])
            and not contains_latin(span["text"])
        ]
    else:
        raise ValueError(f"Unsupported language: {language}")

    groups = group_spans_by_y(selected)

    lines = []

    for group in groups:
        ordered = sorted(
            group,
            key=lambda span: span["x0"],
            reverse=(language == "ar"),
        )

        text = normalize_spaces(
            " ".join(span["text"] for span in ordered)
        )

        lines.append(
            {
                "page": None,
                "y": group[0]["y0"],
                "text": text,
            }
        )

    return lines


def collect_language_lines(
    doc: pymupdf.Document,
    language: str,
) -> list[dict]:
    """Collect reconstructed language lines across the whole PDF."""

    all_lines = []

    for page_number, page in enumerate(doc, start=1):
        spans = extract_spans(page)

        lines = reconstruct_language_lines(
            spans,
            language,
        )

        for line in lines:
            line["page"] = page_number
            all_lines.append(line)

    return all_lines


def detect_article_header(
    text: str,
    language: str,
) -> int | None:
    """Detect an article header in the requested language."""

    if language == "en":
        return detect_english_article(text)

    if language == "ar":
        return detect_arabic_article(text)

    raise ValueError(f"Unsupported language: {language}")


def collect_language_article_headers(
    doc: pymupdf.Document,
    language: str,
) -> dict[int, dict]:
    """
    Collect article headers for one language.

    Only the first candidate for each article number is retained.
    """

    lines = collect_language_lines(
        doc,
        language,
    )

    headers: dict[int, dict] = {}

    for line in lines:
        article_number = detect_article_header(
            line["text"],
            language,
        )

        if article_number is None:
            continue

        if not 1 <= article_number <= 1149:
            continue

        if article_number not in headers:
            headers[article_number] = {
                "page": line["page"],
                "y": line["y"],
                "text": line["text"],
            }

    return headers


def line_position(line: dict) -> tuple[int, float]:
    """Return a sortable document position."""

    return (
        line["page"],
        line["y"],
    )


def extract_language_articles(
    doc: pymupdf.Document,
    language: str,
) -> dict[int, str]:
    """
    Extract article text independently for one language.

    Duplicate header detections are ignored after the first valid
    occurrence of an article number.
    """

    lines = collect_language_lines(
        doc,
        language,
    )

    header_positions = []
    seen_articles: set[int] = set()

    for index, line in enumerate(lines):
        article_number = detect_article_header(
            line["text"],
            language,
        )

        if article_number is None:
            continue

        if not 1 <= article_number <= 1149:
            continue

        # Ignore duplicate occurrences such as later references
        # to an already-seen article.
        if article_number in seen_articles:
            continue

        seen_articles.add(article_number)

        header_positions.append(
            (
                index,
                article_number,
            )
        )

    articles: dict[int, str] = {}

    for position, (start_index, article_number) in enumerate(
        header_positions
    ):
        if position + 1 < len(header_positions):
            end_index = header_positions[position + 1][0]
        else:
            end_index = len(lines)

        content_lines = lines[
            start_index + 1 : end_index
        ]

        content = normalize_spaces(
            " ".join(
                line["text"]
                for line in content_lines
            )
        )

        articles[article_number] = content

    return articles


def collect_article_headers(
    doc: pymupdf.Document,
) -> dict[int, dict]:
    """
    Collect article headers using both English and Arabic headers.

    English headers are preferred because they are generally cleaner,
    but Arabic headers are used as a fallback when the English header
    is missing or malformed.
    """
    headers: dict[int, dict] = {}

    for page_number, page in enumerate(doc, start=1):
        spans = extract_spans(page)

        english_lines = reconstruct_english_lines(spans)
        arabic_lines = reconstruct_arabic_lines(spans)

        # First collect English article headers.
        for line in english_lines:
            article_number = detect_english_article(line["text"])

            if article_number is None:
                continue

            if not 1 <= article_number <= 1149:
                continue

            if article_number not in headers:
                headers[article_number] = {
                    "page": page_number,
                    "y": line["y"],
                    "text": line["text"],
                    "language": "en",
                }

        # Then use Arabic headers for articles whose English header
        # was not detected.
        for line in arabic_lines:
            article_number = detect_arabic_article(line["text"])

            if article_number is None:
                continue

            if not 1 <= article_number <= 1149:
                continue

            if article_number not in headers:
                headers[article_number] = {
                    "page": page_number,
                    "y": line["y"],
                    "text": line["text"],
                    "language": "ar",
                }

    return headers


def expected_article_numbers() -> set[int]:
    """
    Return article numbers expected to have actual article headers.

    Repealed articles are intentionally excluded because the PDF
    does not contain individual article text for those ranges.
    """
    return {
        number
        for number in range(1, 1150)
        if not is_repealed(number)
    }


def validate_article_headers(
    headers: dict[int, dict],
) -> None:
    """Validate detected article-header coverage."""

    expected = expected_article_numbers()
    actual = set(headers)

    missing = sorted(expected - actual)

    unexpected = sorted(
        actual - set(range(1, 1150))
    )

    if missing:
        raise ValueError(
            "Unexpected missing article headers: "
            f"{missing}"
        )

    if unexpected:
        raise ValueError(
            "Unexpected article numbers detected: "
            f"{unexpected}"
        )

    repealed = sorted(
        set(range(1, 1150)) - expected
    )

    english_count = sum(
        1
        for header in headers.values()
        if header["language"] == "en"
    )

    arabic_fallback_count = sum(
        1
        for header in headers.values()
        if header["language"] == "ar"
    )

    print(f"Detected article headers: {len(actual)}")
    print(f"English headers: {english_count}")
    print(f"Arabic fallback headers: {arabic_fallback_count}")
    print(f"Expected active article headers: {len(expected)}")
    print(f"Repealed article numbers: {repealed}")

    arabic_fallback_articles = sorted(
        number
        for number, header in headers.items()
        if header["language"] == "ar"
    )

    if arabic_fallback_articles:
        print()
        print("Articles detected through Arabic fallback:")
        print(arabic_fallback_articles)

def build_article_records(
    doc: pymupdf.Document,
    headers: dict[int, dict],
) -> list[dict]:
    """Build structured article records with bilingual text."""

    print("Extracting English article text...")
    articles_en = extract_language_articles(
        doc,
        "en",
    )

    print("Extracting Arabic article text...")
    articles_ar = extract_language_articles(
        doc,
        "ar",
    )

    records = []

    for article_number in range(1, 1150):
        repealed = is_repealed(article_number)

        header = headers.get(article_number)

        source_page = (
            header["page"]
            if header is not None
            else None
        )

        if repealed:
            text_ar = ""
            text_en = ""
        else:
            text_ar = articles_ar.get(
                article_number,
                "",
            )

            text_en = articles_en.get(
                article_number,
                "",
            )

        records.append(
            {
                "article_number": article_number,
                "book": None,
                "chapter": None,
                "section": None,
                "topic": None,
                "text_ar": text_ar,
                "text_en": text_en,
                "is_repealed": repealed,
                "source_page": source_page,
                "citation": (
                    f"Egyptian Civil Code, "
                    f"Article {article_number}"
                ),
            }
        )

    return records



def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    doc = pymupdf.open(PDF_PATH)

    print(f"PDF: {PDF_PATH}")
    print(f"Pages: {len(doc)}")

    headers = collect_article_headers(doc)

    validate_article_headers(headers)

    records = build_article_records(
        doc,
        headers,
    )

    active_records = [
        record
        for record in records
        if not record["is_repealed"]
    ]

    missing_arabic = [
        record["article_number"]
        for record in active_records
        if not record["text_ar"].strip()
    ]

    missing_english = [
        record["article_number"]
        for record in active_records
        if not record["text_en"].strip()
    ]

    print()
    print(
        f"Active articles missing Arabic text: "
        f"{len(missing_arabic)}"
    )

    print(
        f"Active articles missing English text: "
        f"{len(missing_english)}"
    )

    if missing_arabic:
        print("Missing Arabic:", missing_arabic)

    if missing_english:
        print("Missing English:", missing_english)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Records written: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")

    validate_article_headers(headers)

    records = build_article_records(
    doc,
    headers,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Records written: {len(records)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()