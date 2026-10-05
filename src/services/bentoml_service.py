from __future__ import annotations

import os

import bentoml
from dotenv import load_dotenv
from pydantic import BaseModel

from src.rag.api_generator import APIGenerator
from src.rag.context_builder import build_context
from src.rag.generator import Generator
from src.rag.mock_generator import MockGenerator
from src.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from src.rag.retriever import Retriever


load_dotenv()


class AskRequest(BaseModel):
    question: str


def create_generator() -> Generator:
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


@bentoml.service(
    resources={"cpu": "1"},
    traffic={"timeout": 120},
)
class LegalRAGService:

    def __init__(self):
        self.retriever = Retriever()
        self.generator = create_generator()

    @bentoml.api
    def ask(self, request: AskRequest) -> dict:
        results = self.retriever.search(
            request.question,
            k=5,
        )

        context = build_context(results)

        user_prompt = build_prompt(
            request.question,
            context,
        )

        answer = self.generator.generate(
            SYSTEM_PROMPT,
            user_prompt,
        )

        sources = [
            {
                "article_number": result["metadata"]["article_number"],
                "citation": result["metadata"]["citation"],
                "source_page": result["metadata"].get("source_page"),
            }
            for result in results
        ]

        return {
            "answer": answer,
            "sources": sources,
        }