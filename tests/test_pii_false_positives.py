from src.guardrails.pii import redact_pii


CLEAN_ARABIC_CASES = [
    "العقد بين محمد علي وأحمد محمد.",
    "يجوز للمالك التصرف في ملكه وفقاً للقانون.",
    "تنتقل الملكية بالتسجيل وفقاً لأحكام القانون.",
    "يجب تنفيذ العقد بحسن نية.",
    "يحق للدائن مطالبة المدين بالوفاء بالدين.",
    "تنقضي الالتزامات بالأسباب التي يحددها القانون.",
    "يكون البيع صحيحاً إذا توافرت أركانه القانونية.",
    "تسري أحكام التقادم على الحقوق وفقاً للقانون.",
    "المادة 147 تنظم القوة الملزمة للعقد.",
    "المادة 163 تتعلق بالمسؤولية عن الأعمال غير المشروعة.",
]


def test_clean_arabic_false_positive_rate():
    false_positives = 0

    for text in CLEAN_ARABIC_CASES:
        _, detected = redact_pii(text)

        if detected:
            false_positives += 1

    false_positive_rate = false_positives / len(CLEAN_ARABIC_CASES)

    print(f"\nFalse-positive rate: {false_positive_rate:.2%}")

    assert false_positive_rate <= 0.05