from __future__ import annotations

from src.rag.generator import Generator


class MockGenerator(Generator):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "LLM generation is not configured yet.\n\n"
            "The RAG pipeline successfully retrieved legal context. "
            "A production LLM can be connected through the Generator interface."
        )