from __future__ import annotations

import json
from pathlib import Path

QUESTIONS_PATH = Path("data/evaluation/ragas_questions.json")
CORPUS_PATH = Path("data/processed/corpus_raw.json")
OUTPUT_PATH = Path("data/evaluation/ragas_dataset.json")


def main() -> None:
    questions = json.loads(
        QUESTIONS_PATH.read_text(encoding="utf-8")
    )

    corpus = json.loads(
        CORPUS_PATH.read_text(encoding="utf-8")
    )

    articles = {
        item["article_number"]: item
        for item in corpus
    }

    dataset = []

    for item in questions:
        article_number = item["expected_article"]
        article = articles[article_number]

        dataset.append(
            {
                "user_input": item["user_input"],
                "expected_article": article_number,
                "reference": article["text_ar"],
                "reference_context": {
                    "article_number": article_number,
                    "citation": article["citation"],
                    "source_page": article["source_page"],
                },
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(
            dataset,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Created: {OUTPUT_PATH}")
    print(f"Questions: {len(dataset)}")


if __name__ == "__main__":
    main()