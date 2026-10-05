from __future__ import annotations
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.context_builder import build_context
from src.rag.mock_generator import MockGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from src.rag.retriever import Retriever
import os
from langfuse import get_client
from src.rag.api_generator import APIGenerator
from src.rag.mock_generator import MockGenerator


load_dotenv()
langfuse = get_client()

app = FastAPI(
    title="Egyptian Civil Code RAG API",
    version="0.1.0",
)


retriever = Retriever()


def create_generator():
    generator_type = os.getenv("GENERATOR", "mock").lower()

    if generator_type == "api":
        return APIGenerator(
            model=os.environ["LLM_MODEL"],
        )

    if generator_type == "mock":
        return MockGenerator()

    raise ValueError(
        f"Unsupported GENERATOR value: {generator_type}"
    )


generator = create_generator()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Source(BaseModel):
    article_number: int
    citation: str
    source_page: int | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "documents_indexed": retriever.vector_store.index.ntotal,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    with langfuse.start_as_current_observation(
        as_type="span",
        name="rag-ask",
        input={"question": request.question},
    ) as trace:

        # -------------------------
        # Retrieval
        # -------------------------
        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieval",
            input={"question": request.question, "top_k": 5},
        ) as retrieval_span:

            results = retriever.search(request.question, k=5)

            articles = [
                result["metadata"]["article_number"]
                for result in results
            ]

            print("ASK RETRIEVAL:")
            for result in results:
                print(
                    result["metadata"]["article_number"],
                    result.get("score"),
                )

            retrieval_span.update(
                output={
                    "articles": articles,
                    "scores": [
                        result.get("score")
                        for result in results
                    ],
                }
            )

        # -------------------------
        # Prompt construction
        # -------------------------
        context = build_context(results)
        user_prompt = build_prompt(
            request.question,
            context,
        )

        # -------------------------
        # LLM generation
        # -------------------------
        with langfuse.start_as_current_observation(
            as_type="generation",
            name="llm-generation",
            model=os.environ["LLM_MODEL"],
            input={
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": user_prompt,
            },
        ) as generation:

            answer = generator.generate(
                SYSTEM_PROMPT,
                user_prompt,
            )

            generation.update(
                output=answer,
            )

        # -------------------------
        # Final trace output
        # -------------------------
        trace.update(
            output={
                "answer": answer,
                "retrieved_articles": articles,
            }
        )

        sources = [
            Source(
                article_number=result["metadata"]["article_number"],
                citation=result["metadata"]["citation"],
                source_page=result["metadata"].get("source_page"),
            )
            for result in results
        ]

        return AskResponse(
            answer=answer,
            sources=sources,
        )

    @app.on_event("shutdown")
    def shutdown_event():
        langfuse.flush()