from __future__ import annotations

import asyncio
import os

from nlq_validator.integrations.base import BaseLLMIntegration


class GeminiIntegration(BaseLLMIntegration):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. Install it with: pip install 'nlq-validator[gemini]'"
            )
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model or os.environ.get("GEMINI_MODEL")

    def generate_questions(self, system_prompt: str, count: int = 50) -> list[str]:
        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model)
        prompt = self._build_prompt(system_prompt, count)
        response = model.generate_content(prompt)
        text = response.text if response.text else ""
        return self._parse_response(text)

    async def generate_questions_async(self, system_prompt: str, count: int = 50) -> list[str]:
        # google-generativeai does not expose a native async client;
        # run the synchronous call in a thread to avoid blocking the event loop.
        return await asyncio.to_thread(self.generate_questions, system_prompt, count)
