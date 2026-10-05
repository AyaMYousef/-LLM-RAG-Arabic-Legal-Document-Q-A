from src.guardrails.pii import redact_pii


def test_egyptian_phone_redaction():
    text = "اتصل على 01012345678"

    result, detected = redact_pii(text)

    assert "[REDACTED_PHONE]" in result
    assert "01012345678" not in result
    assert "egyptian_phone" in detected


def test_egyptian_national_id_redaction():
    text = "الرقم القومي هو 29801011234567"

    result, detected = redact_pii(text)

    assert "[REDACTED_NATIONAL_ID]" in result
    assert "29801011234567" not in result
    assert "egyptian_national_id" in detected


def test_email_redaction():
    text = "البريد الإلكتروني example@test.com"

    result, detected = redact_pii(text)

    assert "[REDACTED_EMAIL]" in result
    assert "example@test.com" not in result
    assert "email" in detected


def test_arabic_name_is_not_redacted():
    text = "العقد بين محمد علي وأحمد محمد."

    result, detected = redact_pii(text)

    assert result == text
    assert detected == []