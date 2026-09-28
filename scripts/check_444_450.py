import pymupdf
import scripts.prepare_corpus as c

doc = pymupdf.open("data/raw/egyptian_civil_code.pdf")

lines = c.collect_language_lines(doc, "en")

for number in range(444, 451):
    print(f"\n=== ARTICLE {number} ===")

    found = False

    for index, line in enumerate(lines):
        text = line["text"]

        if str(number) in text and (
            "Article" in text or "rticle" in text
        ):
            detected = c.detect_english_article(text)

            print(
                f"index={index}, "
                f"page={line['page']}, "
                f"y={line['y']}, "
                f"detected={detected}, "
                f"text={text!r}"
            )
            found = True

    if not found:
        print("No candidate found.")