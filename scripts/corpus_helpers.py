from __future__ import annotations

import re


BOOK_RE = re.compile(r"^BOOK\s+[IVXLCDM]+$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+$", re.IGNORECASE)
SECTION_RE = re.compile(r"^SECTION\s+[IVXLCDM]+$", re.IGNORECASE)

AR_BOOK_RE = re.compile(r"^الباب\b")
AR_CHAPTER_RE = re.compile(r"^الفصل\b")
AR_SECTION_RE = re.compile(r"^القسم\b")

TOPIC_RE = re.compile(r"^\d+\s*[-.]\s*(.+)$")


def normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace and strip surrounding spaces."""
    return re.sub(r"\s+", " ", text).strip()


def contains_arabic(text: str) -> bool:
    """Return True if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))

def is_topic_heading(line: dict) -> bool:
    """Return True when a line is a bold topic heading."""
    return is_bold_line(line)

def is_bold_line(line: dict) -> bool:
    """
    Return True if at least one span in the line is bold.

    PyMuPDF reports flags=16 for the bold font used by the PDF.
    """
    return any(
        span.get("flags", 0) & 16
        for span in line.get("spans", [])
    )


def extract_page_lines(page, page_number: int) -> list[dict]:
    """
    Extract page lines with position and font metadata.

    The PDF is bilingual and uses separate Arabic and English columns,
    so we preserve span metadata for later topic detection.
    """
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


def extract_hierarchy(doc) -> dict[int, dict]:
    """
    Extract hierarchy metadata and associate it with articles.

    Hierarchy:

        BOOK       -> book
        CHAPTER    -> chapter
        SECTION    -> section
        bold topic row
            -> topic_ar / topic_en

    The PDF contains Arabic and English topic headings in separate
    columns. Topic detection therefore relies on PDF bold formatting
    rather than guessing from ordinary English article text.
    """

    hierarchy_by_article = {}

    current_book = None
    current_chapter = None
    current_section = None

    current_topic_ar = None
    current_topic_en = None

    pending_structure = None

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        lines = extract_page_lines(page, page_number)

        for index, line in enumerate(lines):
            text = line["text"]

            # ---------------------------------------------------------
            # Article header
            # ---------------------------------------------------------
            article_match = re.match(
                r"^Article\s*(\d+)\b",
                text,
                re.IGNORECASE,
            )

            if article_match:
                article_number = int(article_match.group(1))

                hierarchy_by_article[article_number] = {
                    "book": current_book,
                    "chapter": current_chapter,
                    "section": current_section,
                    "topic_ar": current_topic_ar,
                    "topic_en": current_topic_en,
                }

                pending_structure = None
                continue

            # ---------------------------------------------------------
            # Structural headings
            # ---------------------------------------------------------
            if BOOK_RE.fullmatch(text):
                pending_structure = "book"
                continue

            if CHAPTER_RE.fullmatch(text):
                pending_structure = "chapter"
                continue

            if SECTION_RE.fullmatch(text):
                pending_structure = "section"
                continue

            # ---------------------------------------------------------
            # Value belonging to BOOK / CHAPTER / SECTION
            # ---------------------------------------------------------
            if pending_structure is not None:
                if contains_arabic(text):
                    continue

                if BOOK_RE.fullmatch(text):
                    continue

                if CHAPTER_RE.fullmatch(text):
                    continue

                if SECTION_RE.fullmatch(text):
                    continue

                if pending_structure == "book":
                    current_book = text

                elif pending_structure == "chapter":
                    current_chapter = text

                elif pending_structure == "section":
                    current_section = text

                pending_structure = None
                continue

            # ---------------------------------------------------------
            # Topic detection
            #
            # Topics in the PDF are bold and appear in separate
            # Arabic / English columns.
            #
            # We deliberately DO NOT use:
            #
            #   English-only text
            #   previous Arabic line
            #   Article within next N lines
            #
            # because that incorrectly classified article sentences
            # as topics.
            # ---------------------------------------------------------
            # Ignore Arabic structural labels such as:
            # الباب الأول
            # الفصل الأول
            # القسم الأول
            if (
                AR_BOOK_RE.match(text)
                or AR_CHAPTER_RE.match(text)
                or AR_SECTION_RE.match(text)
            ):
                continue

            if not is_bold_line(line):
                continue

            # Ignore Article headers that happen to be bold.
            if re.match(r"^Article\s*\d+\b", text, re.IGNORECASE):
                continue

            # Ignore BOOK / CHAPTER / SECTION labels.
            if (
                BOOK_RE.fullmatch(text)
                or CHAPTER_RE.fullmatch(text)
                or SECTION_RE.fullmatch(text)
            ):
                continue

            # Ignore structural numbered headings for now.
            #
            # The actual bilingual topic text will be extracted from
            # the bold Arabic / English cells.
            if TOPIC_RE.fullmatch(text):
                topic_text = TOPIC_RE.fullmatch(text).group(1).strip()
            else:
                topic_text = text

            if not topic_text:
                continue

            # ---------------------------------------------------------
            # Determine column.
            #
            # English text in this PDF is on the left side.
            # Arabic text is on the right side.
            #
            # We use the line's x coordinate rather than the language
            # alone because a topic row can contain both languages.
            # ---------------------------------------------------------
            if contains_arabic(topic_text) and not contains_latin(topic_text):
                current_topic_ar = topic_text

            elif contains_latin(topic_text) and not contains_arabic(topic_text):
                current_topic_en = topic_text

            else:
                # Mixed Arabic/English bold line.
                #
                # Keep it for now only when the line itself is clearly
                # a topic row. We will refine column splitting after
                # inspecting the actual span coordinates.
                if line["x"] >= 250:
                    current_topic_ar = topic_text
                else:
                    current_topic_en = topic_text

    return hierarchy_by_article


def contains_latin(text: str) -> bool:
    """Return True if text contains Latin alphabet characters."""
    return bool(re.search(r"[A-Za-z]", text))
