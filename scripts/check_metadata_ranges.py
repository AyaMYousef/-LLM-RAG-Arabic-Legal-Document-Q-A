import json

with open(
    "data/processed/corpus_raw.json",
    encoding="utf-8",
) as file:
    corpus = json.load(file)

article_map = {
    article["article_number"]: article
    for article in corpus
}

for number in [
    802,
    803,
    804,
    805,
    806,
    1028,
    1029,
    1030,
    1031,
    1032,
]:
    article = article_map[number]

    print(
        f"{number}: "
        f"section={article.get('section')!r} | "
        f"topic_en={article.get('topic_en')!r} | "
        f"topic_ar={article.get('topic_ar')!r}"
    )