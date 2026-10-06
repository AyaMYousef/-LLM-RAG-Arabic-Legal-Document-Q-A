from __future__ import annotations

from sentence_transformers import SentenceTransformer

from src.ingestion.vector_store import FAISSVectorStore

MODEL_NAME = "intfloat/multilingual-e5-base"

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
    model = SentenceTransformer(MODEL_NAME)

  
    store = FAISSVectorStore.load("data/processed/vector_store_e5")

    for test in TEST_QUERIES:
        query = f"query: {test['query']}"

        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )

        results = store.search(embedding, k=5)

        print("=" * 80)
        print(f"QUERY: {test['query']}")
        print(f"EXPECTED ARTICLE: {test['expected_article']}")
        print("-" * 80)

        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]

            print(
                f"{rank}. "
                f"score={result['score']:.4f} | "
                f"article={metadata['article_number']} | "
                f"{metadata['citation']}"
            )


if __name__ == "__main__":
    main()