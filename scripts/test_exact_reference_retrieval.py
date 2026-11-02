
from sentence_transformers import SentenceTransformer

from src.ingestion.vector_store import FAISSVectorStore

MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_DIR = "data/processed/vector_store_e5"

TESTS = [
    {
        "name": "Semantic query",
        "query": "ما هو عقد البيع؟",
        "expected_article": 418,
    },
    {
        "name": "Exact article reference",
        "query": "ما الذي ينص عليه القانون في المادة 418؟",
        "expected_article": 418,
    },
    {
        "name": "Semantic query - ownership",
        "query": "ما هي حقوق مالك الشيء؟",
        "expected_article": 802,
    },
    {
        "name": "Exact article reference - ownership",
        "query": "ما الذي تنص عليه المادة 802؟",
        "expected_article": 802,
    },
]


def main():
    model = SentenceTransformer(MODEL_NAME)
    store = FAISSVectorStore.load(INDEX_DIR)

    for test in TESTS:
        embedding = model.encode(
            [test["query"]],
            normalize_embeddings=True,
        )

        results = store.search(embedding, k=5)

        articles = [
            result["metadata"]["article_number"]
            for result in results
        ]

        expected = test["expected_article"]

        rank = (
            articles.index(expected) + 1
            if expected in articles
            else None
        )

        print("\n" + "-" * 60)
        print("Test:", test["name"])
        print("Query:", test["query"])
        print("Expected:", expected)
        print("Retrieved:", articles)
        print("Expected rank:", rank)


if __name__ == "__main__":
    main()
