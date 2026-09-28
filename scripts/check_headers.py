import pymupdf
import scripts.prepare_corpus as c

doc = pymupdf.open("data/raw/egyptian_civil_code.pdf")
headers = c.collect_article_headers(doc)

numbers = [
    54, 601, 627, 660, 703,
    901, 966, 993, 1005, 1022,
    1088, 1092, 1118,
]

for number in numbers:
    if number in headers:
        header = headers[number]
        print(
            f"Article {number}: "
            f"page={header['page']}, "
            f"y={header['y']}, "
            f"language={header['language']}, "
            f"text={header['text']!r}"
        )
    else:
        print(f"Article {number}: NOT FOUND")