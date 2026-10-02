from __future__ import annotations

import json
import random
import re
from pathlib import Path


INPUT_PATH = Path("data/processed/corpus_raw.json")
REPORT_PATH = Path("reports/corpus_validation.txt")

FIRST_ARTICLE = 1
LAST_ARTICLE = 1149

REPEALED_RANGES = (
    range(54, 81),
    range(389, 418),
)
ARABIC_TEXT_EXCEPTIONS = {
    1022: "Arabic text is absent from the source PDF.",
}


REQUIRED_FIELDS = {
    "article_number",
    "book",
    "chapter",
    "section",
    "topic_ar",
    "topic_en",
    "text_ar",
    "text_en",
    "is_repealed",
    "source_page",
    "citation",
}

RANDOM_SAMPLE_SIZE = 20
RANDOM_SEED = 42
MAX_ARTICLE_CHARS = 20_000


def is_repealed(article_number: int) -> bool:
    """Return True if an article belongs to a known repealed range."""
    return any(
        article_number in repealed_range
        for repealed_range in REPEALED_RANGES
    )


def contains_arabic(text: str) -> bool:
    """Return True if text contains Arabic characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def contains_latin(text: str) -> bool:
    """Return True if text contains Latin characters."""
    return bool(re.search(r"[A-Za-z]", text))


def load_corpus() -> list[dict]:
    """Load the generated corpus JSON."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {INPUT_PATH}"
        )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Corpus root must be a JSON list."
        )

    return data


def validate_schema(
    records: list[dict],
    errors: list[str],
) -> None:
    """Validate record structure and required fields."""

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(
                f"Record {index} is not a JSON object."
            )
            continue

        missing = REQUIRED_FIELDS - set(record)

        if missing:
            errors.append(
                f"Record {index} "
                f"(article {record.get('article_number')}) "
                f"missing fields: {sorted(missing)}"
            )

    article_numbers = [
        record.get("article_number")
        for record in records
        if isinstance(record, dict)
    ]

    non_integer = [
        number
        for number in article_numbers
        if not isinstance(number, int)
    ]

    if non_integer:
        errors.append(
            f"Non-integer article numbers: {non_integer[:20]}"
        )

    repealed_types = [
        record.get("article_number")
        for record in records
        if not isinstance(record.get("is_repealed"), bool)
    ]

    if repealed_types:
        errors.append(
            "Records with non-boolean is_repealed: "
            f"{repealed_types[:20]}"
        )


def validate_article_coverage(
    records: list[dict],
    errors: list[str],
) -> None:
    """Validate that all expected article numbers exist exactly once."""

    article_numbers = [
        record["article_number"]
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("article_number"), int)
    ]

    duplicates = sorted(
        {
            number
            for number in article_numbers
            if article_numbers.count(number) > 1
        }
    )

    if duplicates:
        errors.append(
            f"Duplicate article numbers: {duplicates}"
        )

    expected = set(
        range(
            FIRST_ARTICLE,
            LAST_ARTICLE + 1,
        )
    )

    actual = set(article_numbers)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing:
        errors.append(
            f"Missing article numbers: {missing}"
        )

    if unexpected:
        errors.append(
            f"Unexpected article numbers: {unexpected}"
        )


def validate_repealed_flags(
    records: list[dict],
    errors: list[str],
) -> None:
    """Validate repealed article flags against configured ranges."""

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        expected = is_repealed(article_number)
        actual = record.get("is_repealed")

        if actual != expected:
            errors.append(
                f"Article {article_number}: "
                f"is_repealed={actual}, "
                f"expected={expected}"
            )


def validate_article_text(
    records: list[dict],
    errors: list[str],
    warnings: list[str],
) -> None:
    """
    Validate Arabic and English text.

    Repealed articles are allowed to have empty text.
    Active articles must contain both languages,
    except for documented source exceptions.
    """

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        if record.get("is_repealed"):
            continue

        text_ar = record.get("text_ar")
        text_en = record.get("text_en")

        # Arabic source exception
        if article_number in ARABIC_TEXT_EXCEPTIONS:
            warnings.append(
                f"Article {article_number}: "
                f"{ARABIC_TEXT_EXCEPTIONS[article_number]}"
            )
        else:
            if not isinstance(text_ar, str) or not text_ar.strip():
                errors.append(
                    f"Article {article_number}: "
                    "missing Arabic text"
                )

            if isinstance(text_ar, str):
                if not contains_arabic(text_ar):
                    errors.append(
                        f"Article {article_number}: "
                        "Arabic text contains no Arabic characters"
                    )

                if len(text_ar) > MAX_ARTICLE_CHARS:
                    errors.append(
                        f"Article {article_number}: "
                        f"Arabic text exceeds {MAX_ARTICLE_CHARS} "
                        "characters"
                    )

        # English text is still required
        if not isinstance(text_en, str) or not text_en.strip():
            errors.append(
                f"Article {article_number}: "
                "missing English text"
            )

        if isinstance(text_en, str):
            if not contains_latin(text_en):
                errors.append(
                    f"Article {article_number}: "
                    "English text contains no Latin characters"
                )

            if len(text_en) > MAX_ARTICLE_CHARS:
                errors.append(
                    f"Article {article_number}: "
                    f"English text exceeds {MAX_ARTICLE_CHARS} "
                    "characters"
                )

def validate_hierarchy(
    records: list[dict],
    warnings: list[str],
) -> None:
    """
    Check hierarchy metadata.

    Not every article is expected to have all hierarchy fields,
    especially at document transitions, so these are warnings.
    """

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        if record.get("is_repealed"):
            continue

        missing = []

        for field in (
            "book",
            "chapter",
            "section",
        ):
            value = record.get(field)

            if value is None or not str(value).strip():
                missing.append(field)

        if missing:
            warnings.append(
                f"Article {article_number}: "
                f"missing hierarchy fields {missing}"
            )


def validate_topics(
    records: list[dict],
    warnings: list[str],
) -> None:
    """Check topic metadata without requiring it for every article."""

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        if record.get("is_repealed"):
            continue

        topic_ar = record.get("topic_ar")
        topic_en = record.get("topic_en")

        if not topic_ar and not topic_en:
            warnings.append(
                f"Article {article_number}: "
                "missing both topic_ar and topic_en"
            )


def validate_citations(
    records: list[dict],
    errors: list[str],
) -> None:
    """Validate article citations."""

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        expected = (
            f"Egyptian Civil Code, "
            f"Article {article_number}"
        )

        actual = record.get("citation")

        if actual != expected:
            errors.append(
                f"Article {article_number}: "
                f"invalid citation: {actual!r}"
            )


def validate_source_pages(
    records: list[dict],
    errors: list[str],
) -> None:
    """Validate source page metadata."""

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        if record.get("is_repealed"):
            continue

        source_page = record.get("source_page")

        if not isinstance(source_page, int):
            errors.append(
                f"Article {article_number}: "
                f"invalid source_page: {source_page!r}"
            )
            continue

        if source_page < 1:
            errors.append(
                f"Article {article_number}: "
                f"invalid source_page: {source_page}"
            )


def validate_heading_bleed(
    records: list[dict],
    warnings: list[str],
) -> None:
    """Look for obvious heading text accidentally stored as article text."""

    suspicious_patterns = (
        r"^BOOK\s+[IVXLCDM]+$",
        r"^CHAPTER\s+[IVXLCDM]+$",
        r"^SECTION\s+[IVXLCDM]+$",
        r"^Article\s+\d+$",
    )

    compiled = [
        re.compile(
            pattern,
            re.IGNORECASE,
        )
        for pattern in suspicious_patterns
    ]

    for record in records:
        article_number = record.get("article_number")

        if not isinstance(article_number, int):
            continue

        for field in ("text_ar", "text_en"):
            value = record.get(field)

            if not isinstance(value, str):
                continue

            first_line = value.strip().splitlines()[0] \
                if value.strip() else ""

            if any(
                pattern.fullmatch(first_line)
                for pattern in compiled
            ):
                warnings.append(
                    f"Article {article_number}: "
                    f"possible heading bleed in {field}"
                )


def validate_random_sample(
    records: list[dict],
    warnings: list[str],
) -> list[str]:
    """Inspect a reproducible random sample of active articles."""

    active_records = [
        record
        for record in records
        if isinstance(record, dict)
        and not record.get("is_repealed")
    ]

    if not active_records:
        warnings.append(
            "No active articles available for random sampling."
        )
        return []

    sample_size = min(
        RANDOM_SAMPLE_SIZE,
        len(active_records),
    )

    random.seed(RANDOM_SEED)

    sample = random.sample(
        active_records,
        sample_size,
    )

    sample.sort(
        key=lambda record: record["article_number"]
    )

    results = []

    for record in sample:
        article_number = record["article_number"]
        source_page = record.get("source_page")

        ar_length = len(
            record.get("text_ar", "")
        )

        en_length = len(
            record.get("text_en", "")
        )

        results.append(
            f"Article {article_number} "
            f"→ PDF page {source_page} "
            f"(AR chars={ar_length}, "
            f"EN chars={en_length})"
        )

    return results


def build_report(
    records: list[dict],
    errors: list[str],
    warnings: list[str],
    sample: list[str],
) -> str:
    """Build the validation report."""

    lines = []

    lines.append("=" * 80)
    lines.append("CORPUS VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("INPUT")
    lines.append("-" * 80)
    lines.append(f"Corpus: {INPUT_PATH}")
    lines.append(f"Records: {len(records)}")
    lines.append("")

    lines.append("EXPECTED SCHEMA")
    lines.append("-" * 80)
    lines.append(
        ", ".join(sorted(REQUIRED_FIELDS))
    )
    lines.append("")

    lines.append("REPEALED RANGES")
    lines.append("-" * 80)
    lines.append("Articles 54–80")
    lines.append("Articles 389–417")
    lines.append("")

    lines.append("RANDOM SAMPLE")
    lines.append("-" * 80)

    if sample:
        lines.extend(sample)
    else:
        lines.append("No sample available.")

    lines.append("")

    lines.append("ERRORS")
    lines.append("-" * 80)

    if errors:
        lines.extend(
            f"[ERROR] {error}"
            for error in errors
        )
    else:
        lines.append("None")

    lines.append("")

    lines.append("WARNINGS")
    lines.append("-" * 80)

    if warnings:
        lines.extend(
            f"[WARNING] {warning}"
            for warning in warnings
        )
    else:
        lines.append("None")

    lines.append("")

    lines.append("=" * 80)

    if errors:
        lines.append("RESULT: FAIL")
    else:
        lines.append("RESULT: PASS")

    lines.append("=" * 80)

    return "\n".join(lines)


def main() -> None:
    records = load_corpus()

    errors: list[str] = []
    warnings: list[str] = []

    validate_schema(
        records,
        errors,
    )

    validate_article_coverage(
        records,
        errors,
    )

    validate_repealed_flags(
        records,
        errors,
    )

    validate_article_text(
        records,
        errors,
        warnings,
    )

    validate_hierarchy(
        records,
        warnings,
    )

    validate_topics(
        records,
        warnings,
    )

    validate_citations(
        records,
        errors,
    )

    validate_source_pages(
        records,
        errors,
    )

    validate_heading_bleed(
        records,
        warnings,
    )

    sample = validate_random_sample(
        records,
        warnings,
    )

    report = build_report(
        records,
        errors,
        warnings,
        sample,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print()
    print(f"Report written to: {REPORT_PATH}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()