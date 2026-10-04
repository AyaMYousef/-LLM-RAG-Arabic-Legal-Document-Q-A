from __future__ import annotations

import json
from pathlib import Path

from src.rag.retriever import Retriever


INPUT_PATH = Path("data/evaluation/ragas_dataset.json")
OUTPUT_PATH = Path("data/evaluation/ragas_inputs.json")


def main() -> None:
    dataset = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    retriever = Retriever()

    results = []

    for i, item in enumerate(dataset, start=1):
        question = item["user_input"]

        retrieved = retriever.search(
            question,
            k=5,
        )

        retrieved_contexts = [
            (
                f"{result['metadata']['citation']}\n"
                f"{result['metadata'].get('text_ar', '')}"
            )
            for result in retrieved
        ]

        results.append(
            {
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "reference": item["reference"],
                "expected_article": item["expected_article"],
            }
        )

        print(
            f"[{i:02d}/{len(dataset)}] "
            f"Article {item['expected_article']}"
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(f"Created: {OUTPUT_PATH}")
    print(f"Questions: {len(results)}")


if __name__ == "__main__":
    main()