from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.context_builder import build_context
from src.rag.retriever import Retriever


app = FastAPI(
    title="Egyptian Civil Code RAG API",
    version="0.1.0",
)


retriever = Retriever()


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

    sources = [
        Source(
            article_number=result["metadata"]["article_number"],
            citation=result["metadata"]["citation"],
            source_page=result["metadata"].get("source_page"),
        )
        for result in results
    ]

    return AskResponse(
        answer=(
            "LLM generation is not connected yet. "
            "Retrieved legal context is ready for generation."
        ),
        sources=sources,
    )