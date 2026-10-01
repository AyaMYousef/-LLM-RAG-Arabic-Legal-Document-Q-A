from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

from corpus_helpers import extract_hierarchy, extract_page_lines, is_bold_line

PDF_PATH = Path("data/raw/egyptian_civil_code.pdf")
OUTPUT_PATH = Path("data/processed/corpus_raw.json")

Y_TOLERANCE = 2.0
ARABIC_CODE_START_PAGE = 1
ARABIC_CODE_START_Y = 250.0

ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩",
    "0123456789",
)

# Articles explicitly identified as repealed in the PDF.
#
REPEALED_RANGES = (
    range(54, 81),
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
    text = normalize_spaces(text)
    text = re.sub(r"[()\[\]{}]", " ", text)
    text = normalize_spaces(text)

    match = re.match(
        r"^(?:Article|rticle)\s*(\d+)\b",
        text,
    )
    if not match:
        return None

    return int(match.group(1))

def detect_arabic_article(text: str) -> int | None:
    text = normalize_spaces(text)
    text = re.sub(r"[()\[\]{}]", " ", text)
    text = normalize_spaces(text)

    match = re.match(
        r"^(?:مادة|ما\s*دة|م\s*ادة|ماد\s*ة)\s+([٠-٩0-9][٠-٩0-9\s]*)\b",
        text,
    )

    if not match:
        return None

    digits = normalize_digits(match.group(1))

    # The PDF can split a visually reversed article number
    # into multiple digit groups, e.g. "٠٦ ١" for Article 601.
    # Reverse each group separately, then concatenate them.
    groups = re.findall(r"[0-9]+", digits)

    normalized_number = "".join(
        group[::-1]
        for group in groups
    )

    return int(normalized_number)




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

    Article references can look like genuine headers. Candidates are
    therefore selected in numerical/document order while preserving
    chronological consistency between consecutive article numbers.
    """
    lines = collect_language_lines(doc, language)

    expected = sorted(expected_article_numbers())
    expected_set = set(expected)
    
    candidates: dict[int, list[int]] = {}

    for index, line in enumerate(lines):
        article_number = detect_article_header(
            line["text"],
            language,
        )

        if article_number is None:
            continue

        if article_number not in expected_set:
            continue

        # Ignore the Law of Promulgation on page 1.
        # Its "مادة ١" is not Civil Code Article 1.
        if (
            language == "ar"
            and line["page"] == ARABIC_CODE_START_PAGE
            and line["y"] < ARABIC_CODE_START_Y
        ):
            continue

        candidates.setdefault(article_number, []).append(index)

    selected: dict[int, int] = {}

    for position, article_number in enumerate(expected):
        article_candidates = candidates.get(article_number, [])

        if not article_candidates:
            continue

        previous_article = (
            expected[position - 1]
            if position > 0
            else None
        )

        previous_index = (
            selected.get(previous_article, -1)
            if previous_article is not None
            else -1
        )

        valid_candidates = [
            index
            for index in article_candidates
            if index > previous_index
        ]

        if not valid_candidates:
            continue

        exact_candidates = []

        for index in valid_candidates:
            text = normalize_spaces(lines[index]["text"])

            cleaned = re.sub(
                r"[()\[\]{}]",
                " ",
                text,
            )

            cleaned = normalize_spaces(cleaned)

            if language == "en":
                if re.fullmatch(
                    rf"(?:Article|rticle)\s*{article_number}\s*[.]?",
                    cleaned,
                ):
                    exact_candidates.append(index)

            elif language == "ar":
                match = re.match(
                r"^(?:مادة|ما\s*دة|م\s*ادة|ماد\s*ة)\s+([٠-٩0-9][٠-٩0-9\s]*)\b",
                cleaned,
            )

                if match:
                    digits = normalize_digits(match.group(1))
                    digits = re.sub(r"\s+", "", digits)

                    reversed_number = int(digits[::-1])
                    direct_number = int(digits)


                    if article_number in {
                        reversed_number,
                        direct_number,
                    }:
                        if len(cleaned) <= 30:
                            exact_candidates.append(index)

        if exact_candidates:
            selected[article_number] = exact_candidates[0]
        else:
            selected[article_number] = valid_candidates[0]

    header_positions = sorted(
        (
            index,
            article_number,
        )
        for article_number, index in selected.items()
    )

    articles: dict[int, str] = {}

     # Targeted fallback for Arabic article bodies whose Arabic header
    # is missing or malformed in the PDF extraction.
   # arabic_fallbacks = {
    #    1022: (101, 102),
    #}

    for position, (
        start_index,
        article_number,
    ) in enumerate(header_positions):

        if position + 1 < len(header_positions):
            end_index = header_positions[position + 1][0]
        else:
            end_index = len(lines)

        content_lines = lines[
            start_index + 1:end_index
        ]

        content = normalize_spaces(
            " ".join(
                line["text"]
                for line in content_lines
                if not is_bold_line(line)
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
    hierarchy: dict[int, dict],
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

        metadata = hierarchy.get(
            article_number,
            {},
        )

        records.append(
            {
                "article_number": article_number,
                "book": metadata.get("book"),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
                "topic_ar": metadata.get("topic_ar"),
                "topic_en": metadata.get("topic_en"),
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

def extract_page_lines(page, page_number: int) -> list[dict]:
    """Extract English / Arabic lines with their page coordinates."""

    lines = []

    for block in page.get_text("dict")["blocks"]:
        if "lines" not in block:
            continue

        for line in block["lines"]:
            spans = line["spans"]

            if not spans:
                continue

            text = normalize_spaces(
                "".join(span["text"] for span in spans)
            )

            if not text:
                continue

            x = spans[0]["bbox"][0]
            y = spans[0]["bbox"][1]

            lines.append(
            {
                "page": page_number,
                "x": x,
                "y": y,
                "text": text,
                "spans": [
                    {
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "flags": span["flags"],
                    }
                    for span in spans
                ],
            }
        )

    lines.sort(key=lambda item: (item["y"], item["x"]))

    return lines

def debug_hierarchy_pages():
    doc = pymupdf.open("data/raw/egyptian_civil_code.pdf")

    for page_number in range(159, 162):
        print(f"\n{'=' * 100}")
        print(f"PAGE {page_number}")
        print(f"{'=' * 100}")

        lines = extract_page_lines(
            doc[page_number - 1],
            page_number,
        )

        for line in lines:
            print(
                f"Y={line['y']:8.2f} "
                f"X={line['x']:8.2f} "
                f"{line['text']!r}"
            )

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

    for page in doc:
        lines = extract_page_lines(page, page.number + 1)

        for line in lines:
            if (
                "The Effects of a Contract" in line["text"]
                or "over these implements." in line["text"]
            ):
                print("\n" + "=" * 80)
                print("PAGE:", line["page"])
                print("TEXT:", repr(line["text"]))
                print("SPANS:")

                for span in line["spans"]:
                    print(
                        "  ",
                        "TEXT:", repr(span["text"]),
                        "FONT:", span["font"],
                        "SIZE:", span["size"],
                        "FLAGS:", span["flags"],
                    )

    print(f"PDF: {PDF_PATH}")
    print(f"Pages: {len(doc)}")

    debug_hierarchy_pages()

    hierarchy = extract_hierarchy(doc)
    print("\nHIERARCHY CHECK")
    print("=" * 100)

    for article_number, metadata in hierarchy.items():
            if (
                metadata["topic_ar"] is None
                or metadata["topic_en"] is None
            ):
                print(
                    f"Article {article_number}: "
                    f"book={metadata['book']!r}, "
                    f"chapter={metadata['chapter']!r}, "
                    f"section={metadata['section']!r}, "
                    f"topic_ar={metadata['topic_ar']!r}, "
                    f"topic_en={metadata['topic_en']!r}"
                )


            print("\nTOPIC VALUES")
            print("=" * 100)

            topics = {}

            for article_number, metadata in hierarchy.items():
                topic_en = metadata["topic_en"]
                topics.setdefault(topic_en, []).append(article_number)

            for topic_en, articles in topics.items():
                print(f"{topic_en!r}: {len(articles)} articles")


            print("\nSAMPLE HIERARCHY")
            print("=" * 100)

            for article_number in [
                89,
                90,
                145,
                146,
                147,
                148,
                418,
                1099,
                1100,
                1101,
                1102,
                1103,
                1104,
            ]:
                print(
                    article_number,
                    "->",
                    hierarchy.get(article_number),
                )


            headers = collect_article_headers(doc)

            validate_article_headers(headers)

            records = build_article_records(
                doc,
                headers,
                hierarchy,
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


if __name__ == "__main__":
    
    main()