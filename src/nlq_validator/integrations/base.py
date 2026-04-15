import re
from abc import ABC, abstractmethod
from pathlib import Path


class BaseLLMIntegration(ABC):
    _PROMPT_TEMPLATE = (
        "You are generating training data for a query validator.\n"
        "Given the system prompt below, generate {count} diverse, realistic questions "
        "that users might ask this assistant.\n\n"
        "System prompt: {system_prompt}\n\n"
        "Rules:\n"
        "- Return exactly {count} questions\n"
        "- One question per line\n"
        "- No numbering, bullets, or extra text\n"
        "- Vary question length, phrasing, and complexity"
    )

    @abstractmethod
    def generate_questions(self, system_prompt: str, count: int = 50) -> list[str]:
        """Call the LLM synchronously and return `count` sample questions."""

    @abstractmethod
    async def generate_questions_async(self, system_prompt: str, count: int = 50) -> list[str]:
        """Call the LLM asynchronously and return `count` sample questions."""

    def generate_and_save(
        self,
        system_prompt: str,
        path: str | Path,
        count: int = 50,
    ) -> list[str]:
        """Generate questions synchronously and save them to a file."""
        questions = self.generate_questions(system_prompt, count)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(questions), encoding="utf-8")
        return questions

    async def generate_and_save_async(
        self,
        system_prompt: str,
        path: str | Path,
        count: int = 50,
    ) -> list[str]:
        """Generate questions asynchronously and save them to a file."""
        questions = await self.generate_questions_async(system_prompt, count)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(questions), encoding="utf-8")
        return questions

    def _build_prompt(self, system_prompt: str, count: int) -> str:
        return self._PROMPT_TEMPLATE.format(system_prompt=system_prompt, count=count)

    def _parse_response(self, text: str) -> list[str]:
        """Strip blank lines and common list prefixes from raw LLM response."""
        questions = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d]+[.):\-]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()
            if line:
                questions.append(line)
        return questions
