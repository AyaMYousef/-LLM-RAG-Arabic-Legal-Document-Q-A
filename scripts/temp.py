import json

with open("data/processed/corpus_raw.json", encoding="utf-8") as f:
    corpus = json.load(f)

for number in [1, 1021, 1022, 1023]:
    article = next(
        x for x in corpus
        if x["article_number"] == number
    )

    print()
    print("=" * 100)
    print(f"ARTICLE {number}")
    print("=" * 100)

    print("\n--- ARABIC ---")
    print(article["text_ar"])

    print("\n--- ENGLISH ---")
    print(article["text_en"])