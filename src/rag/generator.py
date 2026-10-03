from __future__ import annotations

from abc import ABC, abstractmethod


class Generator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer from the supplied prompts."""
        raise NotImplementedError