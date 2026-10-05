from __future__ import annotations

import os

from openai import OpenAI

from src.rag.generator import Generator


class APIGenerator(Generator):
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model

        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL"),
            api_key=api_key or os.getenv("LLM_API_KEY", "EMPTY"),
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
            max_tokens=300,
        )

        print("LLM RESPONSE:")
        print(response)

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(f"LLM returned an empty response. Full response: {response}")

        return content