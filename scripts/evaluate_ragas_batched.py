from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from evaluate_ragas import (
    E5RagasEmbeddings,
    create_evaluator_llm,
    load_samples,
)
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

BATCH_SIZE = 5
INITIAL_WAIT_SECONDS = 60
MAX_RETRIES = 4

OUTPUT_FILE = Path("data/evaluation/ragas_results_batched.json")
CHECKPOINT_FILE = Path("data/evaluation/ragas_checkpoint.json")


def save_checkpoint(rows, completed):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(
        json.dumps(
            {"completed_batches": completed, "results": rows},
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )


def main():
    samples = load_samples()
    embeddings = E5RagasEmbeddings()
    llm = create_evaluator_llm()

    metrics = [
        Faithfulness(llm=llm),
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=embeddings),
    ]

    total = len(samples)
    rows = []
    completed_batches = 0

    print(f"Total questions: {total}")
    print(f"Batch size: {BATCH_SIZE}")

    for start in range(0, total, BATCH_SIZE):
        batch_number = start // BATCH_SIZE + 1
        batch = samples[start : start + BATCH_SIZE]
        print(
            f"\nBatch {batch_number}: "
            f"questions {start + 1}-{start + len(batch)} of {total}"
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = evaluate(
                    dataset=EvaluationDataset(samples=batch),
                    metrics=metrics,
                    embeddings=embeddings,
                    raise_exceptions=True,
                )

                batch_df = result.to_pandas()
                batch_rows = batch_df.to_dict(orient="records")

                # Keep results from the completed batch.
                rows.extend(batch_rows)
                completed_batches += 1
                save_checkpoint(rows, completed_batches)

                score_columns = [
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                    "answer_relevancy",
                ]

                for column in score_columns:
                    if column in batch_df:
                        valid = pd.to_numeric(batch_df[column], errors="coerce")
                        print(
                            f"  {column}: "
                            f"{valid.notna().sum()}/{len(batch_df)} "
                            "numeric scores"
                        )

                print(f"Checkpoint saved: {len(rows)}/{total} rows")
                break

            except Exception as exc:
                message = str(exc).lower()
                is_rate_limit = any(
                    term in message
                    for term in (
                        "rate limit",
                        "ratelimit",
                        "429",
                        "too many requests",
                    )
                )

                if not is_rate_limit or attempt == MAX_RETRIES:
                    save_checkpoint(rows, completed_batches)
                    print("Evaluation stopped. Completed batches were saved.")
                    raise

                wait = INITIAL_WAIT_SECONDS * (2**attempt)
                print(
                    f"Rate limit detected. Waiting {wait}s "
                    f"before retry {attempt + 1}/{MAX_RETRIES}."
                )
                time.sleep(wait)

        # Pause between successful batches to reduce request bursts.
        if start + BATCH_SIZE < total:
            print("Waiting 60 seconds before the next batch...")
            time.sleep(60)

    df = pd.DataFrame(rows)
    score_columns = [
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    ]

    aggregate = {}
    for column in score_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        aggregate[column] = float(values.mean()) if values.notna().any() else None

    output = {
        "num_samples": len(samples),
        "num_results": len(df),
        "metrics": aggregate,
        "results": rows,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )

    print("\nEvaluation finished.")
    print(f"Questions: {len(samples)}")
    print(f"Results: {OUTPUT_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print("Aggregate metrics:", json.dumps(aggregate, indent=2))

    if len(df) != len(samples):
        raise RuntimeError(f"Incomplete evaluation: {len(df)}/{len(samples)} rows.")


if __name__ == "__main__":
    main()
