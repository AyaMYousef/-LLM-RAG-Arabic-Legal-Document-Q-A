from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator


class Generator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate an answer from the supplied prompts."""
        raise NotImplementedError

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Iterator[str]:
        """Generate an answer as a stream of text chunks."""
        yield self.generate(system_prompt, user_prompt)