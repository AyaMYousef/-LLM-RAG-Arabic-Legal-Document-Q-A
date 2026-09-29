import pymupdf

from scripts.corpus_helpers import extract_hierarchy

PDF_PATH = "data/raw/egyptian_civil_code.pdf"

doc = pymupdf.open(PDF_PATH)

hierarchy = extract_hierarchy(doc)

for article_number in [89, 90, 147, 418]:
    print("=" * 80)
    print(f"ARTICLE {article_number}")
    print(hierarchy.get(article_number))