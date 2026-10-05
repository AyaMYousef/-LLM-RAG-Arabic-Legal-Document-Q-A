import re


# Egyptian mobile numbers:
# 010xxxxxxxx, 011xxxxxxxx, 012xxxxxxxx, 015xxxxxxxx
EGYPTIAN_PHONE_PATTERN = re.compile(
    r"\b01[0125]\d{8}\b"
)

# Egyptian national ID:
# 14 consecutive digits.
EGYPTIAN_NATIONAL_ID_PATTERN = re.compile(
    r"\b\d{14}\b"
)

# Email addresses.
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


def redact_pii(text: str) -> tuple[str, list[str]]:
    """
    Detect and redact high-confidence PII from generated text.

    Returns:
        redacted_text: Text with detected PII replaced.
        detected_types: Types of PII detected.
    """

    detected_types: list[str] = []

    def replace_phone(match: re.Match) -> str:
        detected_types.append("egyptian_phone")
        return "[REDACTED_PHONE]"

    def replace_national_id(match: re.Match) -> str:
        detected_types.append("egyptian_national_id")
        return "[REDACTED_NATIONAL_ID]"

    def replace_email(match: re.Match) -> str:
        detected_types.append("email")
        return "[REDACTED_EMAIL]"

    text = EGYPTIAN_PHONE_PATTERN.sub(
        replace_phone,
        text,
    )

    text = EGYPTIAN_NATIONAL_ID_PATTERN.sub(
        replace_national_id,
        text,
    )

    text = EMAIL_PATTERN.sub(
        replace_email,
        text,
    )

    return text, detected_types