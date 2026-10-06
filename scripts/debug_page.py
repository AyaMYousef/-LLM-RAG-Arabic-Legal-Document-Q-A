import pymupdf
from corpus_helpers import extract_page_lines

doc = pymupdf.open("data/raw/egyptian_civil_code.pdf")

lines = extract_page_lines(doc[0], 1)

for line in lines:
    if 150 <= line["y"] <= 750:
        is_bold = any(
            span.get("flags", 0) & 16
            for span in line["spans"]
        )

        print(
            f'Y={line["y"]:7.2f} '
            f'X={line["x"]:7.2f} '
            f'B={is_bold} '
            f'TEXT={line["text"]!r}'
        )