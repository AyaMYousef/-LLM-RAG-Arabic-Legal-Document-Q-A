"""Retrieval evaluation script with sliced metrics and failure analysis.

Computes Recall@1/3/5 and MRR broken down by:
  - query language  (ar / en)
  - query type      (semantic / article_number_lookup)

Failed queries (RR = 0) are saved to reports/retrieval_failures.json for
iterative improvement. Retrieval metrics are kept separate from generation
metrics so each part of the system can be evaluated independently.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --retriever hybrid
    python scripts/evaluate_retrieval.py --k 5 --output reports/my_run.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer

from src.ingestion.vector_store import FAISSVectorStore

MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_DIR = "data/processed/vector_store_e5"
FAILURES_PATH = Path("reports/retrieval_failures.json")
SLICE_REPORT_PATH = Path("reports/retrieval_slice_report.json")

# ---------------------------------------------------------------------------
# Test queries – annotated with language and query type
#   query_lang : "ar" | "en"
#   query_type : "semantic"            – conceptual / paraphrase queries
#                "article_number_lookup" – queries that name or imply a
#                                          specific article number, making
#                                          exact-term match important
# ---------------------------------------------------------------------------
TEST_QUERIES: list[dict[str, Any]] = [
    # ── Arabic / Semantic ────────────────────────────────────────────────────
    {
        "query": "ماذا يحدث إذا لم يوجد نص تشريعي يمكن تطبيقه؟",
        "expected_article": 1,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما الذي يحدث عند عدم وجود نص قانوني ينطبق على المسألة؟",
        "expected_article": 1,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما هي حالات إلغاء النص التشريعي؟",
        "expected_article": 2,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "متى يكون استعمال الحق غير مشروع؟",
        "expected_article": 5,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "متى يعتبر استعمال الحق مخالفا للقانون؟",
        "expected_article": 5,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "متى تبدأ الشخصية القانونية للإنسان؟",
        "expected_article": 29,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "متى تنتهي الشخصية القانونية للإنسان؟",
        "expected_article": 29,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "كيف تثبت الولادة والوفاة؟",
        "expected_article": 30,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "كيف يتم إثبات الولادة إذا لم توجد سجلات رسمية؟",
        "expected_article": 30,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "متى يمكن طلب وقف الاعتداء على حقوق الشخصية والتعويض؟",
        "expected_article": 50,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ماذا يعني أن العقد شريعة المتعاقدين؟",
        "expected_article": 147,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "متى يجوز للقاضي رد الالتزام التعاقدي المرهق إلى الحد المعقول؟",
        "expected_article": 147,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "كيف يجب تنفيذ العقد؟",
        "expected_article": 148,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما الذي يقتضيه تنفيذ العقد بحسن نية؟",
        "expected_article": 148,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما هي الالتزامات التي تنشأ مباشرة عن القانون وحده؟",
        "expected_article": 198,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما هو عقد البيع؟",
        "expected_article": 418,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "بماذا يضمن البائع المشتري بالنسبة إلى المبيع؟",
        "expected_article": 439,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما هي حقوق مالك الشيء؟",
        "expected_article": 802,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما هي القواعد التي تحكم تحديد الورثة وأنصبائهم وانتقال أموال التركة؟",
        "expected_article": 875,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    {
        "query": "ما هو الرهن الرسمي؟",
        "expected_article": 1030,
        "query_lang": "ar",
        "query_type": "semantic",
    },
    # ── Arabic / Article-number lookup ───────────────────────────────────────
    {
        "query": "نص المادة 147 من القانون المدني المصري",
        "expected_article": 147,
        "query_lang": "ar",
        "query_type": "article_number_lookup",
    },
    {
        "query": "نص المادة 1030 رهن رسمي",
        "expected_article": 1030,
        "query_lang": "ar",
        "query_type": "article_number_lookup",
    },
    {
        "query": "المادة 418 عقد البيع",
        "expected_article": 418,
        "query_lang": "ar",
        "query_type": "article_number_lookup",
    },
    # ── English / Semantic ───────────────────────────────────────────────────
    {
        "query": "What happens when there is no applicable legislative text?",
        "expected_article": 1,
        "query_lang": "en",
        "query_type": "semantic",
    },
    {
        "query": "When is the exercise of a right considered unlawful?",
        "expected_article": 5,
        "query_lang": "en",
        "query_type": "semantic",
    },
    {
        "query": "When does legal personality begin for a natural person?",
        "expected_article": 29,
        "query_lang": "en",
        "query_type": "semantic",
    },
    {
        "query": "What does it mean that the contract is the law of the parties?",
        "expected_article": 147,
        "query_lang": "en",
        "query_type": "semantic",
    },
    {
        "query": "How must a contract be performed?",
        "expected_article": 148,
        "query_lang": "en",
        "query_type": "semantic",
    },
    {
        "query": "What is a contract of sale?",
        "expected_article": 418,
        "query_lang": "en",
        "query_type": "semantic",
    },
    {
        "query": "What are the rights of the owner of a thing?",
        "expected_article": 802,
        "query_lang": "en",
        "query_type": "semantic",
    },
    # ── English / Article-number lookup ─────────────────────────────────────
    {
        "query": "Article 147 Egyptian Civil Code contract",
        "expected_article": 147,
        "query_lang": "en",
        "query_type": "article_number_lookup",
    },
    {
        "query": "Article 1030 formal mortgage pledge",
        "expected_article": 1030,
        "query_lang": "en",
        "query_type": "article_number_lookup",
    },
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved_articles: list[int], expected: int) -> float:
    """Return 1/rank of the expected article, or 0.0 if not retrieved."""
    for rank, article in enumerate(retrieved_articles, start=1):
        if article == expected:
            return 1.0 / rank
    return 0.0


def compute_metrics(
    results: list[dict[str, Any]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Aggregate Recall@k and MRR over a list of per-query result dicts."""
    if not results:
        return {}

    n = len(results)
    metrics: dict[str, float] = {}

    for k in k_values:
        hits = sum(1 for r in results if r[f"hit@{k}"])
        metrics[f"recall@{k}"] = hits / n

    metrics["mrr"] = sum(r["rr"] for r in results) / n
    metrics["n_queries"] = n
    return metrics


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    if not metrics:
        return
    n = int(metrics.get("n_queries", 0))
    print(f"\n  [{label}]  n={n}")
    for k in (1, 3, 5):
        key = f"recall@{k}"
        if key in metrics:
            print(f"    Recall@{k}: {metrics[key]:.4f}")
    if "mrr" in metrics:
        print(f"    MRR:       {metrics['mrr']:.4f}")


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def build_retriever(retriever_type: str):
    """Return (encode_fn, search_fn) based on retriever type."""
    print(f"\nLoading E5 model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading FAISS index from {INDEX_DIR}...")
    store = FAISSVectorStore.load(INDEX_DIR)

    if retriever_type == "dense":
        def encode(query: str) -> Any:
            return model.encode([query], normalize_embeddings=True)

        def search(embedding: Any, k: int) -> list[dict[str, Any]]:
            return store.search(embedding, k=k)

        return encode, search

    elif retriever_type == "hybrid":
        try:
            from src.ingestion.bm25_store import BM25Store
        except ImportError:
            print(
                "[ERROR] BM25Store not found. "
                "Run with --retriever dense or implement src/ingestion/bm25_store.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

        bm25_path = Path("data/processed/bm25_store")
        if not bm25_path.exists():
            print(
                f"[ERROR] BM25 index not found at {bm25_path}. "
                "Build it with scripts/build_bm25_index.py first.",
                file=sys.stderr,
            )
            sys.exit(1)

        bm25 = BM25Store.load(bm25_path)

        def encode(query: str) -> Any:  # type: ignore[misc]
            return model.encode([query], normalize_embeddings=True)

        def search(embedding: Any, k: int) -> list[dict[str, Any]]:  # type: ignore[misc]
            return bm25.hybrid_search(store, embedding, query_text=_current_query[0], k=k)

        return encode, search

    else:
        print(f"[ERROR] Unknown retriever type: {retriever_type}", file=sys.stderr)
        sys.exit(1)


# Mutable container so the closure in hybrid search can capture the current query
_current_query: list[str] = [""]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    retriever_type: str = "dense",
    k_max: int = 5,
    failures_path: Path = FAILURES_PATH,
    slice_report_path: Path = SLICE_REPORT_PATH,
) -> None:
    k_values = tuple(k for k in (1, 3, 5) if k <= k_max)

    print("=" * 80)
    print("RETRIEVAL EVALUATION")
    print(f"Retriever : {retriever_type}")
    print(f"Queries   : {len(TEST_QUERIES)}")
    print(f"k (max)   : {k_max}")
    print("=" * 80)

    # ── NOTE ON METRIC SEPARATION ──────────────────────────────────────────
    # These metrics cover retrieval only (dense / hybrid vector search).
    # Generation quality metrics (faithfulness, answer relevancy, citation
    # accuracy) are evaluated separately via scripts/evaluate_ragas.py and
    # logged to MLflow.  Keeping them separate allows each sub-system to be
    # debugged and improved independently.
    # ───────────────────────────────────────────────────────────────────────

    encode, search = build_retriever(retriever_type)

    all_results: list[dict[str, Any]] = []

    print("\n" + "-" * 80)
    print("PER-QUERY RESULTS")
    print("-" * 80)

    for item in TEST_QUERIES:
        query = item["query"]
        expected = item["expected_article"]
        q_lang = item["query_lang"]
        q_type = item["query_type"]

        _current_query[0] = query

        embedding = encode(f"query: {query}")
        results = search(embedding, k=k_max)

        retrieved_articles = [
            r["metadata"]["article_number"] for r in results
        ]

        hit_k: dict[str, bool] = {}
        for k in k_values:
            hit_k[f"hit@{k}"] = expected in retrieved_articles[:k]

        rr = reciprocal_rank(retrieved_articles, expected)
        rank = (
            retrieved_articles.index(expected) + 1
            if expected in retrieved_articles
            else None
        )

        row: dict[str, Any] = {
            "query": query,
            "query_lang": q_lang,
            "query_type": q_type,
            "expected_article": expected,
            "retrieved_articles": retrieved_articles,
            "rank": rank,
            "rr": rr,
            **hit_k,
        }
        all_results.append(row)

        status = "✓" if hit_k.get("hit@5") else "✗"
        print(
            f"\n{status} [{q_lang.upper()} / {q_type}]"
            f"  EXPECTED: Article {expected}  |  RANK: {rank}"
        )
        print(f"  Q: {query}")
        print(f"  Retrieved: {retrieved_articles}")

    # ── AGGREGATE METRICS ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("RETRIEVAL METRICS  (dense retrieval only — generation metrics are separate)")
    print("=" * 80)

    overall = compute_metrics(all_results, k_values)
    print_metrics("OVERALL", overall)

    # Slice by language
    print("\n── By Query Language ──")
    by_lang: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_lang[r["query_lang"]].append(r)
    for lang, rows in sorted(by_lang.items()):
        print_metrics(f"lang={lang}", compute_metrics(rows, k_values))

    # Slice by query type
    print("\n── By Query Type ──")
    by_type: dict[str, list] = defaultdict(list)
    for r in all_results:
        by_type[r["query_type"]].append(r)
    for qtype, rows in sorted(by_type.items()):
        print_metrics(f"type={qtype}", compute_metrics(rows, k_values))

    # Slice by language × type
    print("\n── By Language × Type ──")
    by_lang_type: dict[str, list] = defaultdict(list)
    for r in all_results:
        key = f"{r['query_lang']}_{r['query_type']}"
        by_lang_type[key].append(r)
    for key, rows in sorted(by_lang_type.items()):
        print_metrics(key, compute_metrics(rows, k_values))

    # ── FAILURE ANALYSIS ───────────────────────────────────────────────────
    failures = [r for r in all_results if r["rr"] == 0.0]

    print("\n" + "=" * 80)
    print(f"FAILURE ANALYSIS  ({len(failures)} / {len(all_results)} queries failed)")
    print("=" * 80)

    if failures:
        print("\nFailed queries:")
        for f in failures:
            print(
                f"  ✗ [{f['query_lang'].upper()} / {f['query_type']}]"
                f"  Article {f['expected_article']}: {f['query']}"
            )
            print(f"    Retrieved: {f['retrieved_articles']}")
    else:
        print("\n  All queries succeeded at k=5.")

    # ── SAVE REPORTS ───────────────────────────────────────────────────────
    run_ts = datetime.now(UTC).isoformat()

    failures_path.parent.mkdir(parents=True, exist_ok=True)

    failure_records = [
        {
            "run_timestamp": run_ts,
            "retriever": retriever_type,
            **f,
        }
        for f in failures
    ]

    with failures_path.open("w", encoding="utf-8") as fh:
        json.dump(failure_records, fh, ensure_ascii=False, indent=2)

    print(f"\nFailure log saved → {failures_path}")

    # Slice report
    slice_report = {
        "run_timestamp": run_ts,
        "retriever": retriever_type,
        "k_max": k_max,
        "overall": overall,
        "by_language": {
            lang: compute_metrics(rows, k_values)
            for lang, rows in by_lang.items()
        },
        "by_query_type": {
            qtype: compute_metrics(rows, k_values)
            for qtype, rows in by_type.items()
        },
        "by_language_and_type": {
            key: compute_metrics(rows, k_values)
            for key, rows in by_lang_type.items()
        },
        "n_failures": len(failures),
        "failure_rate": len(failures) / len(all_results) if all_results else 0.0,
    }

    with slice_report_path.open("w", encoding="utf-8") as fh:
        json.dump(slice_report, fh, ensure_ascii=False, indent=2)

    print(f"Slice report saved → {slice_report_path}")
    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality with sliced metrics and failure logging."
    )
    parser.add_argument(
        "--retriever",
        choices=["dense", "hybrid"],
        default="dense",
        help="Retriever type to evaluate (default: dense).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Maximum k for Recall@k (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SLICE_REPORT_PATH,
        help=f"Path for slice report JSON (default: {SLICE_REPORT_PATH}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        retriever_type=args.retriever,
        k_max=args.k,
        slice_report_path=args.output,
    )
