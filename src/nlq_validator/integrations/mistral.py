from __future__ import annotations

import asyncio
import os

from nlq_validator.integrations.base import BaseLLMIntegration


class MistralIntegration(BaseLLMIntegration):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            from mistralai import Mistral  # noqa: F401
        except ImportError:
            raise ImportError(
                "mistralai package is required. Install it with: pip install 'nlq-validator[mistral]'"
            )
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._model = model or os.environ.get("MISTRAL_MODEL")

    def generate_questions(self, system_prompt: str, count: int = 50) -> list[str]:
        from mistralai import Mistral

        client = Mistral(api_key=self._api_key)
        prompt = self._build_prompt(system_prompt, count)
        response = client.chat.complete(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
        )
        text = response.choices[0].message.content if response.choices else ""
        return self._parse_response(text)

    async def generate_questions_async(self, system_prompt: str, count: int = 50) -> list[str]:
        # mistralai's async client requires an explicit event loop context;
        # wrapping in asyncio.to_thread is the safest cross-version approach.
        return await asyncio.to_thread(self.generate_questions, system_prompt, count)
