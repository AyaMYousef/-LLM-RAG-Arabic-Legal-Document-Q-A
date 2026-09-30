from __future__ import annotations

import re


BOOK_RE = re.compile(r"^BOOK\s+[IVXLCDM]+$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLCDM]+$", re.IGNORECASE)
SECTION_RE = re.compile(r"^SECTION\s+[IVXLCDM]+$", re.IGNORECASE)
TOPIC_RE = re.compile(
    r"^\d+\s*\.\s*(.+)$"
)

SUBHEADING_RE = re.compile(
    r"^\d+\s*-\s*(.+)$"
)


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

    Hierarchy:

        BOOK       -> book
        CHAPTER    -> chapter
        SECTION    -> section
        numbered heading -> structural subtopic
        plain English heading before articles -> topic
    """

    hierarchy_by_article = {}

    current_book = None
    current_chapter = None
    current_section = None
    current_topic = None

    pending_structure = None
    pending_plain_heading = None

    for page_index, page in enumerate(doc):
        page_number = page_index + 1
        lines = extract_page_lines(page, page_number)

        for index, line in enumerate(lines):
            text = line["text"]

            # ---------------------------------------------------------
            # ARTICLE
            # ---------------------------------------------------------
            article_match = re.match(
                r"^Article\s*(\d+)\b",
                text,
                re.IGNORECASE,
            )

            if article_match:
                article_number = int(article_match.group(1))

                # A plain English heading immediately before the
                # article becomes the topic.
                if pending_plain_heading is not None:
                    current_topic = pending_plain_heading
                    pending_plain_heading = None

                hierarchy_by_article[article_number] = {
                    "book": current_book,
                    "chapter": current_chapter,
                    "section": current_section,
                    "topic": current_topic,
                }

                pending_structure = None
                continue

            # ---------------------------------------------------------
            # BOOK
            # ---------------------------------------------------------
            if BOOK_RE.fullmatch(text):
                pending_structure = "book"
                continue

            # ---------------------------------------------------------
            # CHAPTER
            # ---------------------------------------------------------
            if CHAPTER_RE.fullmatch(text):
                pending_structure = "chapter"
                continue

            # ---------------------------------------------------------
            # SECTION
            # ---------------------------------------------------------
            if SECTION_RE.fullmatch(text):
                pending_structure = "section"
                continue

            # ---------------------------------------------------------
            # STRUCTURAL TITLE
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

                # English title following BOOK / CHAPTER / SECTION
                if pending_structure == "book":
                    current_book = text

                elif pending_structure == "chapter":
                    current_chapter = text

                elif pending_structure == "section":
                    current_section = text

                pending_structure = None
                continue

            # ---------------------------------------------------------
            # NUMBERED SUBHEADING
            # ---------------------------------------------------------
            topic_match = TOPIC_RE.fullmatch(text)

            if topic_match:
                current_topic = topic_match.group(1).strip()
                continue

            if SUBHEADING_RE.fullmatch(text):
                continue

            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # PLAIN ENGLISH TOPIC HEADING
            # ---------------------------------------------------------
            if (
                not contains_arabic(text)
                and not text.lower().startswith("article")
            ):
                previous_line = (
                    lines[index - 1]
                    if index > 0
                    else None
                )

                if (
                    previous_line
                    and contains_arabic(previous_line["text"])
                    and abs(line["y"] - previous_line["y"]) < 5
                ):
                    valid_heading = True

                    for future_index in range(
                        index + 1,
                        min(index + 5, len(lines)),
                    ):
                        future_text = lines[future_index]["text"]

                        if re.match(
                            r"^Article\s*\d+\b",
                            future_text,
                            re.IGNORECASE,
                        ):
                            if valid_heading:
                                pending_plain_heading = text
                            break

                        if looks_like_arabic_article_number(
                            future_text
                        ):
                            continue

                        # Anything else means this is not an
                        # article heading.
                        valid_heading = False
                        break
                        if looks_like_arabic_article_number(future_text):
                            continue

                        # Another English line means this is likely
                        # normal article text rather than a heading.
                        if (
                            not contains_arabic(future_text)
                            and not future_text.lower().startswith("article")
                        ):
                            break

    return hierarchy_by_article



def looks_like_arabic_article_number(text: str) -> bool:
    """Return True for Arabic article-number/header lines."""
    text = normalize_spaces(text)

    if "مادة" in text:
        return True

    if re.fullmatch(r"[٠-٩0-9\s]+", text):
        return True

    return False