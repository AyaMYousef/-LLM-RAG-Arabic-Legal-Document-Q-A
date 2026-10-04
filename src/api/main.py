from __future__ import annotations
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.context_builder import build_context
from src.rag.mock_generator import MockGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from src.rag.retriever import Retriever
import os

from src.rag.api_generator import APIGenerator
from src.rag.mock_generator import MockGenerator


load_dotenv()

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
    results = retriever.search(request.question, k=5)

    context = build_context(results)
    user_prompt = build_prompt(request.question, context)

    answer = generator.generate(
        SYSTEM_PROMPT,
        user_prompt,
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