
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.ingestion.vector_store import FAISSVectorStore

MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_DIR = "data/processed/vector_store_e5"
OUTPUT_FILE = "data/processed/retrieval_failures.json"

# Keep these test queries consistent with evaluate_retrieval.py.
TEST_QUERIES: list[dict[str, Any]] = [
    {"query": "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟", "expected_article": 1},
    {"query": "ما الذي يحدث عند عدم وجود نص قانوني ينطبق على المسألة؟", "expected_article": 1},
    {"query": "ما هي حالات إلغاء النص التشريعي؟", "expected_article": 2},
    {"query": "متى يكون استعمال الحق غير مشروع؟", "expected_article": 5},
    {"query": "متى يعتبر استعمال الحق مخالفا للقانون؟", "expected_article": 5},
    {"query": "متى تبدأ الشخصية القانونية للإنسان؟", "expected_article": 29},
    {"query": "متى تنتهي الشخصية القانونية للإنسان؟", "expected_article": 29},
    {"query": "كيف تثبت الولادة والوفاة؟", "expected_article": 30},
    {"query": "كيف يتم إثبات الولادة إذا لم توجد سجلات رسمية؟", "expected_article": 30},
    {"query": "متى يمكن طلب وقف الاعتداء على حقوق الشخصية والتعويض؟", "expected_article": 50},
    {"query": "ماذا يعني أن العقد شريعة المتعاقدين؟", "expected_article": 147},
    {"query": "متى يجوز للقاضي رد الالتزام التعاقدي المرهق إلى الحد المعقول؟", "expected_article": 147},
    {"query": "كيف يجب تنفيذ العقد؟", "expected_article": 148},
    {"query": "ما الذي يقتضيه تنفيذ العقد بحسن نية؟", "expected_article": 148},
    {"query": "ما هي الالتزامات التي تنشأ مباشرة عن القانون وحده؟", "expected_article": 198},
    {"query": "ما هو عقد البيع؟", "expected_article": 418},
    {"query": "بماذا يضمن البائع المشتري بالنسبة إلى المبيع؟", "expected_article": 439},
    {"query": "ما هي حقوق مالك الشيء؟", "expected_article": 802},
    {"query": "ما هي القواعد التي تحكم تحديد الورثة وأنصبائهم وانتقال أموال التركة؟", "expected_article": 875},
    {"query": "ما هو الرهن الرسمي؟", "expected_article": 1030},
]


def get_article_number(result: dict[str, Any]) -> Any:
    return result.get("metadata", {}).get("article_number")


def get_article_text(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata", {})

    for key in (
        "text",
        "content",
        "article_text",
        "article_content",
        "chunk_text",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value

    # Some vector-store implementations return text outside metadata.
    for key in ("text", "content", "document"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value

    return None


def get_score(result: dict[str, Any]) -> Any:
    for key in ("score", "similarity", "distance"):
        if key in result:
            return result[key]
    return None


def main() -> None:
    print("Loading embedding model...")
    model = SentenceTransformer(
    MODEL_NAME,
    device="cpu",
)

    print("Loading FAISS index...")
    store = FAISSVectorStore.load(INDEX_DIR)

    failures = []
    all_results = []

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected_article"]

        # Preserve the existing baseline behavior.
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )

        results = store.search(embedding, k=5)
        retrieved = [get_article_number(r) for r in results]

        rank = (
            retrieved.index(expected) + 1
            if expected in retrieved
            else None
        )

        record = {
            "query": query,
            "expected_article": expected,
            "expected_rank": rank,
            "hit_at_5": rank is not None,
            "results": [
                {
                    "rank": position,
                    "article_number": get_article_number(result),
                    "score": get_score(result),
                    "text": get_article_text(result),
                }
                for position, result in enumerate(results, start=1)
            ],
        }

        all_results.append(record)

        if rank is None:
            failures.append(record)

        status = f"rank {rank}" if rank is not None else "MISS"
        print(f"\n[{status}] Expected Article {expected}")
        print(f"Query: {query}")
        print(f"Retrieved: {retrieved}")

        if rank is None:
            for result in record["results"]:
                print(
                    f"\n  Rank {result['rank']} | "
                    f"Article {result['article_number']} | "
                    f"Score: {result['score']}"
                )
                text = result["text"]
                if text:
                    print(f"  Text: {text[:500]}")
                else:
                    print("  Text unavailable in returned result.")

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": MODEL_NAME,
        "index_dir": INDEX_DIR,
        "total_queries": len(TEST_QUERIES),
        "failed_queries": len(failures),
        "recall_at_5": (
            (len(TEST_QUERIES) - len(failures)) / len(TEST_QUERIES)
        ),
        "failures": failures,
        "all_results": all_results,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"Total queries: {len(TEST_QUERIES)}")
    print(f"Failed at top 5: {len(failures)}")
    print(f"Recall@5: {report['recall_at_5']:.2%}")
    print(f"Report saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
