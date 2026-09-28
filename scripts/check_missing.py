import pymupdf
import scripts.prepare_corpus as c

doc = pymupdf.open("data/raw/egyptian_civil_code.pdf")

targets_en = [277, 714, 746, 898, 908, 993, 1090]
targets_ar = [54, 601, 627, 660, 703, 966, 1005, 1022, 1088, 1092, 1118]

print("=== ENGLISH CANDIDATES ===")

english_lines = c.collect_language_lines(doc, "en")

for target in targets_en:
    print(f"\n--- Article {target} ---")

    found = False
    for line in english_lines:
        text = line["text"]
        if str(target) in text:
            print(
                f"page={line['page']}, "
                f"y={line['y']}, "
                f"text={text!r}"
            )
            found = True

    if not found:
        print("No English line containing this number found.")

print("\n=== ARABIC CANDIDATES ===")

arabic_lines = c.collect_language_lines(doc, "ar")

for target in targets_ar:
    print(f"\n--- Article {target} ---")

    # Arabic PDF digits are visually reversed.
    target_reversed = str(target)[::-1]

    found = False
    for line in arabic_lines:
        text = line["text"]

        if (
            target_reversed in c.normalize_digits(text)
            or str(target) in c.normalize_digits(text)
        ):
            print(
                f"page={line['page']}, "
                f"y={line['y']}, "
                f"text={text!r}"
            )
            found = True

    if not found:
        print("No Arabic candidate containing this number found.")