from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.rag.context_builder import build_context
from src.rag.mock_generator import MockGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from src.rag.retriever import Retriever


app = FastAPI(
    title="Egyptian Civil Code RAG API",
    version="0.1.0",
)


retriever = Retriever()
generator = MockGenerator()


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