from __future__ import annotations

from src.ingestion.embedder import Embedder
from src.ingestion.vector_store import FAISSVectorStore


TEST_QUERIES = [
    {
        "query": "What happens when there is no applicable legal provision?",
        "expected_article": 1,
    },
    {
        "query": "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟",
        "expected_article": 1,
    },
]


def main() -> None:
    embedder = Embedder()
    store = FAISSVectorStore.load("data/processed/vector_store")

    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected_article"]

        embedding = embedder.encode([query])
        results = store.search(embedding, k=5)

        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"EXPECTED ARTICLE: {expected}")
        print("-" * 80)

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(
                f"{rank}. "
                f"score={result['score']:.4f} | "
                f"article={metadata['article_number']} | "
                f"{metadata['citation']}"
            )

        retrieved_articles = [
            result["metadata"]["article_number"]
            for result in results
        ]

        print(
            f"Retrieved expected article: "
            f"{expected in retrieved_articles}"
        )


if __name__ == "__main__":
    main()