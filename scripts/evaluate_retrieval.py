from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

from src.ingestion.vector_store import FAISSVectorStore


MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_DIR = "data/processed/vector_store_e5"

TEST_QUERIES: list[dict[str, Any]] = [
    {
        "query": "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟",
        "expected_article": 1,
    },
    {
        "query": "ما الذي يحدث عند عدم وجود نص قانوني ينطبق على المسألة؟",
        "expected_article": 1,
    },
    {
        "query": "ما هي حالات إلغاء النص التشريعي؟",
        "expected_article": 2,
    },
    {
        "query": "متى يكون استعمال الحق غير مشروع؟",
        "expected_article": 5,
    },
    {
        "query": "متى يعتبر استعمال الحق مخالفا للقانون؟",
        "expected_article": 5,
    },
    {
        "query": "متى تبدأ الشخصية القانونية للإنسان؟",
        "expected_article": 29,
    },
    {
        "query": "متى تنتهي الشخصية القانونية للإنسان؟",
        "expected_article": 29,
    },
    {
        "query": "كيف تثبت الولادة والوفاة؟",
        "expected_article": 30,
    },
    {
        "query": "كيف يتم إثبات الولادة إذا لم توجد سجلات رسمية؟",
        "expected_article": 30,
    },
    {
        "query": "متى يمكن طلب وقف الاعتداء على حقوق الشخصية والتعويض؟",
        "expected_article": 50,
    },
    {
        "query": "ماذا يعني أن العقد شريعة المتعاقدين؟",
        "expected_article": 147,
    },
    {
        "query": "متى يجوز للقاضي رد الالتزام التعاقدي المرهق إلى الحد المعقول؟",
        "expected_article": 147,
    },
    {
        "query": "كيف يجب تنفيذ العقد؟",
        "expected_article": 148,
    },
    {
        "query": "ما الذي يقتضيه تنفيذ العقد بحسن نية؟",
        "expected_article": 148,
    },
    {
        "query": "ما هي الالتزامات التي تنشأ مباشرة عن القانون وحده؟",
        "expected_article": 198,
    },
    {
        "query": "ما هو عقد البيع؟",
        "expected_article": 418,
    },
    {
        "query": "بماذا يضمن البائع المشتري بالنسبة إلى المبيع؟",
        "expected_article": 439,
    },
    {
        "query": "ما هي حقوق مالك الشيء؟",
        "expected_article": 802,
    },
    {
        "query": "ما هي القواعد التي تحكم تحديد الورثة وأنصبائهم وانتقال أموال التركة؟",
        "expected_article": 875,
    },
    {
        "query": "ما هو الرهن الرسمي؟",
        "expected_article": 1030,
    },
]


def reciprocal_rank(
    retrieved_articles: list[int],
    expected_article: int,
) -> float:
    """Return reciprocal rank of the expected article."""

    for rank, article in enumerate(retrieved_articles, start=1):
        if article == expected_article:
            return 1.0 / rank

    return 0.0


def main() -> None:
    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print("=" * 80)

    print("\nLoading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Loading E5 index...")
    store = FAISSVectorStore.load(INDEX_DIR)

    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks = []

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected_article"]

        query_text = f"query: {query}"

        embedding = model.encode(
            [query_text],
            normalize_embeddings=True,
        )

        results = store.search(
            embedding,
            k=5,
        )

        retrieved_articles = [
            result["metadata"]["article_number"]
            for result in results
        ]

        hit_1 = expected in retrieved_articles[:1]
        hit_3 = expected in retrieved_articles[:3]
        hit_5 = expected in retrieved_articles[:5]

        hits_at_1 += int(hit_1)
        hits_at_3 += int(hit_3)
        hits_at_5 += int(hit_5)

        rr = reciprocal_rank(
            retrieved_articles,
            expected,
        )

        reciprocal_ranks.append(rr)

        rank = (
            retrieved_articles.index(expected) + 1
            if expected in retrieved_articles
            else None
        )

        print("\n" + "-" * 80)
        print(f"QUERY: {query}")
        print(f"EXPECTED: Article {expected}")
        print(f"RETRIEVED: {retrieved_articles}")
        print(f"EXPECTED RANK: {rank}")
        print(f"RECIPROCAL RANK: {rr:.4f}")

    total = len(TEST_QUERIES)

    recall_at_1 = hits_at_1 / total
    recall_at_3 = hits_at_3 / total
    recall_at_5 = hits_at_5 / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Queries:    {total}")
    print(f"Recall@1:   {recall_at_1:.4f}")
    print(f"Recall@3:   {recall_at_3:.4f}")
    print(f"Recall@5:   {recall_at_5:.4f}")
    print(f"MRR:        {mrr:.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()