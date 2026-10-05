from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from sentence_transformers import SentenceTransformer


load_dotenv()

INPUT_FILE = Path("data/evaluation/ragas_responses.json")
OUTPUT_FILE = Path("data/evaluation/ragas_results.json")


class E5RagasEmbeddings(BaseRagasEmbeddings):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
    ):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
        ).tolist()

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self.model.encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
        ).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return self.embed_documents(texts)


def load_samples() -> list[SingleTurnSample]:
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        SingleTurnSample(
            user_input=item["user_input"],
            response=item["response"],
            retrieved_contexts=item["retrieved_contexts"],
            reference=item["reference"],
        )
        for item in data
    ]


def create_evaluator_llm():
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )

    return llm_factory(
        model=os.environ["LLM_MODEL"],
        provider="openai",
        client=client,
    )


def main() -> None:
    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    samples = load_samples()

    print(f"Loaded samples: {len(samples)}")

    evaluator_llm = create_evaluator_llm()

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
    ]

    dataset = EvaluationDataset(samples=samples)

    print()
    print("Starting RAGAS evaluation...")
    print("Evaluator model:", os.environ["LLM_MODEL"])
    print()

    embeddings = E5RagasEmbeddings()

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        embeddings=embeddings,
    )

    print()
    print("=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    print(result)

    result_df = result.to_pandas()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "num_samples": len(samples),
        "metrics": {
            "faithfulness": float(
                result_df["faithfulness"].mean()
            ),
            "context_precision": float(
                result_df["context_precision"].mean()
            ),
            "context_recall": float(
                result_df["context_recall"].mean()
            ),
        },
        "results": result_df.to_dict(orient="records"),
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()