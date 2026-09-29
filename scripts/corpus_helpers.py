from __future__ import annotations

import re


BOOK_RE = re.compile(r"^BOOK\s+[IVXLCDM]+$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+$", re.IGNORECASE)
SECTION_RE = re.compile(r"^SECTION\s+[IVXLCDM]+$", re.IGNORECASE)
TOPIC_RE = re.compile(r"^\d+\.\s*(.+)$")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


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
                }
            )

    lines.sort(key=lambda item: (item["y"], item["x"]))

    return lines


def extract_hierarchy(doc) -> dict[int, dict]:
    """
    Extract hierarchy metadata and associate it with articles.

    Target hierarchy:

        BOOK       -> book
        CHAPTER    -> chapter
        Section    -> section
        numbered heading -> topic
    """

    hierarchy_by_article = {}

    current_book = None
    current_chapter = None
    current_section = None
    current_topic = None

    pending_structure = None

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        lines = extract_page_lines(page, page_number)

        for line in lines:
            text = line["text"]

            # --------------------------------------------------
            # ARTICLE
            # --------------------------------------------------
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
                    "topic": current_topic,
                }

                pending_structure = None
                continue

            # --------------------------------------------------
            # BOOK
            # --------------------------------------------------
            if BOOK_RE.fullmatch(text):
                pending_structure = "book"
                continue

            # --------------------------------------------------
            # CHAPTER
            # --------------------------------------------------
            if CHAPTER_RE.fullmatch(text):
                pending_structure = "chapter"
                continue

            # --------------------------------------------------
            # SECTION
            # --------------------------------------------------
            if SECTION_RE.fullmatch(text):
                pending_structure = "section"
                continue

            # --------------------------------------------------
            # If we are waiting for a structural title,
            # accept only a clean English heading.
            # --------------------------------------------------
            if pending_structure is not None:

                if contains_arabic(text):
                    continue

                if text.lower().startswith("article"):
                    pending_structure = None
                    continue

                if BOOK_RE.fullmatch(text):
                    continue

                if CHAPTER_RE.fullmatch(text):
                    continue

                if SECTION_RE.fullmatch(text):
                    continue

                # Numbered headings are topics, not structural titles.
                if TOPIC_RE.fullmatch(text):
                    pending_structure = None
                    current_topic = normalize_spaces(
                        TOPIC_RE.fullmatch(text).group(1)
                    )
                    continue

                if pending_structure == "book":
                    current_book = text

                elif pending_structure == "chapter":
                    current_chapter = text

                elif pending_structure == "section":
                    current_section = text

                pending_structure = None
                continue

            # --------------------------------------------------
            # TOPIC
            # --------------------------------------------------
            topic_match = TOPIC_RE.fullmatch(text)

            if topic_match:
                current_topic = normalize_spaces(
                    topic_match.group(1)
                )
                continue

    return hierarchy_by_article