from __future__ import annotations

import os

from nlq_validator.integrations.base import BaseLLMIntegration


class ChatGPTIntegration(BaseLLMIntegration):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise ImportError(
                "openai package is required. Install it with: pip install 'nlq-validator[openai]'"
            )
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model or os.environ.get("OPENAI_MODEL")

    def generate_questions(self, system_prompt: str, count: int = 50) -> list[str]:
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        prompt = self._build_prompt(system_prompt, count)
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        text = response.choices[0].message.content or ""
        return self._parse_response(text)

    async def generate_questions_async(self, system_prompt: str, count: int = 50) -> list[str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self._api_key)
        prompt = self._build_prompt(system_prompt, count)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        text = response.choices[0].message.content or ""
        return self._parse_response(text)
