from __future__ import annotations

import json
from pathlib import Path

import requests

INPUT_PATH = Path("data/evaluation/ragas_inputs.json")
OUTPUT_PATH = Path("data/evaluation/ragas_responses.json")

API_URL = "http://127.0.0.1:8000/ask"


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    total = len(questions)

    for i, item in enumerate(questions, start=1):
        question = item["user_input"]

        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=120,
        )
        response.raise_for_status()

        data = response.json()

        results.append(
            {
                "user_input": question,
                "response": data["answer"],
                "retrieved_contexts": item["retrieved_contexts"],
                "reference": item["reference"],
                "expected_article": item["expected_article"],
            }
        )

        print(f"[{i}/{total}] Article {item['expected_article']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print(f"Created: {OUTPUT_PATH}")
    print(f"Questions: {len(results)}")


if __name__ == "__main__":
    main()