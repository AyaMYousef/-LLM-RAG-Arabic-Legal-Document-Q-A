import re

ARTICLE_PATTERNS = [
    r"المادة\s+\d+",
    r"مادة\s+\d+",
    r"Article\s+\d+",
    r"article\s+\d+",
]


def validate_output(
    answer: str,
    retrieved_articles: list[str],
) -> tuple[bool, str | None]:
    """
    Validate that the generated answer is grounded in retrieved legal articles.
    """

    if not answer or not answer.strip():
        return False, "The generated answer is empty."

    answer = answer.strip()

    # The answer should contain an article reference.
    has_article_reference = any(
        re.search(pattern, answer)
        for pattern in ARTICLE_PATTERNS
    )

    if not has_article_reference:
        return (
            False,
            "The generated answer does not contain a legal article citation.",
        )

    # Make sure at least one retrieved article is referenced.
    if retrieved_articles:
        referenced_article = any(
            str(article) in answer
            for article in retrieved_articles
        )

        if not referenced_article:
            return (
                False,
                "The generated answer does not reference the retrieved legal sources.",
            )

    return True, None