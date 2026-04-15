from __future__ import annotations

import os

from nlq_validator.integrations.base import BaseLLMIntegration


class ClaudeIntegration(BaseLLMIntegration):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install it with: pip install 'nlq-validator[anthropic]'"
            )
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model or os.environ.get("ANTHROPIC_MODEL")

    def generate_questions(self, system_prompt: str, count: int = 50) -> list[str]:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        prompt = self._build_prompt(system_prompt, count)
        message = client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text if message.content else ""
        return self._parse_response(text)

    async def generate_questions_async(self, system_prompt: str, count: int = 50) -> list[str]:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        prompt = self._build_prompt(system_prompt, count)
        message = await client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text if message.content else ""
        return self._parse_response(text)
