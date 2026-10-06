import re

LEGAL_KEYWORDS_AR = [
    "قانون",
    "القانون",
    "المادة",
    "العقد",
    "عقد",
    "التزام",
    "التزامات",
    "الحق",
    "الحقوق",
    "الملكية",
    "البيع",
    "الإيجار",
    "التعويض",
    "المسؤولية",
    "الدين",
    "الدينار",
    "الوفاء",
    "بطلان",
    "باطل",
    "فسخ",
    "تقادم",
    "ميراث",
    "رهن",
    "حيازة",
    "ضرر",
    "تعويض",
]

LEGAL_KEYWORDS_EN = [
    "law",
    "article",
    "contract",
    "obligation",
    "right",
    "property",
    "sale",
    "lease",
    "compensation",
    "liability",
    "payment",
    "invalid",
    "termination",
    "prescription",
    "mortgage",
    "possession",
    "damage",
]


def validate_input(question: str) -> tuple[bool, str | None]:
    """
    Validate that the question is suitable for the Egyptian Civil Code RAG system.
    """

    if not question or not question.strip():
        return False, "Question cannot be empty."

    question = question.strip()

    if len(question) > 2000:
        return False, "Question is too long."

    if len(re.findall(r"\S+", question)) < 2:
        return False, "Please provide a complete legal question."

    normalized = question.lower()

    has_arabic = bool(re.search(r"[\u0600-\u06FF]", question))

    keywords = LEGAL_KEYWORDS_AR if has_arabic else LEGAL_KEYWORDS_EN

    if not any(keyword in normalized for keyword in keywords):
        return (
            False,
            "This question does not appear to relate to the Egyptian Civil Code.",
        )

    return True, None