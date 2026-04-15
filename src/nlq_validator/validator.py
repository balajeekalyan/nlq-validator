from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nlq_validator.model import TopicModel

if TYPE_CHECKING:
    from nlq_validator.integrations.base import BaseLLMIntegration


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CalibrationResult:
    in_domain_scores: list[float]
    off_domain_scores: list[float]
    suggested_threshold: float

    def summary(self) -> None:
        """Print a precision/recall/F1 table over a sweep of threshold candidates."""
        all_scores = self.in_domain_scores + self.off_domain_scores
        labels = [1] * len(self.in_domain_scores) + [0] * len(self.off_domain_scores)
        candidates = sorted(set(all_scores))

        header = f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>9}  {'F1':>8}"
        separator = "-" * len(header)
        print(header)
        print(separator)
        for t in candidates:
            preds = [1 if s >= t else 0 for s in all_scores]
            tp = sum(p == 1 and lb == 1 for p, lb in zip(preds, labels))
            fp = sum(p == 1 and lb == 0 for p, lb in zip(preds, labels))
            fn = sum(p == 0 and lb == 1 for p, lb in zip(preds, labels))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            marker = " <-- suggested" if abs(t - self.suggested_threshold) < 1e-9 else ""
            print(f"{t:>10.4f}  {precision:>10.4f}  {recall:>9.4f}  {f1:>8.4f}{marker}")
        print(separator)
        print(f"Suggested threshold: {self.suggested_threshold:.4f}")


class NLQValidator:
    def __init__(
        self,
        model: TopicModel,
        threshold: float = 0.25,
    ) -> None:
        self._model = model
        self._threshold = threshold

    @classmethod
    def from_training_file(
        cls,
        path: str | Path,
        system_prompt: str,
        threshold: float = 0.25,
        embedding_model: str | None = None,
    ) -> "NLQValidator":
        from nlq_validator.loader import FileLoader
        from nlq_validator.trainer import train

        examples = FileLoader.load(path)
        model = train(examples, system_prompt=system_prompt, embedding_model=embedding_model)
        return cls(model, threshold)

    @classmethod
    def from_llm(
        cls,
        integration: "BaseLLMIntegration",
        system_prompt: str,
        count: int = 50,
        threshold: float = 0.25,
        embedding_model: str | None = None,
    ) -> "NLQValidator":
        from nlq_validator.trainer import train

        examples = integration.generate_questions(system_prompt, count)
        model = train(examples, system_prompt=system_prompt, embedding_model=embedding_model)
        return cls(model, threshold)

    @classmethod
    async def from_llm_async(
        cls,
        integration: "BaseLLMIntegration",
        system_prompt: str,
        count: int = 50,
        threshold: float = 0.25,
        embedding_model: str | None = None,
    ) -> "NLQValidator":
        from nlq_validator.trainer import train

        examples = await integration.generate_questions_async(system_prompt, count)
        model = train(examples, system_prompt=system_prompt, embedding_model=embedding_model)
        return cls(model, threshold)

    @classmethod
    def load(
        cls,
        path: str | Path,
        threshold: float = 0.25,
    ) -> "NLQValidator":
        from nlq_validator.persistence import load_model

        model = load_model(path)
        return cls(model, threshold)

    def save(self, path: str | Path) -> None:
        from nlq_validator.persistence import save_model

        save_model(self._model, path)

    def retrain(self, additional_examples: list[str]) -> None:
        """Merge new examples with existing training data and retrain in-place."""
        from nlq_validator.trainer import train

        merged = self._model.user_examples + list(additional_examples)
        self._model = train(
            merged,
            system_prompt=self._model.system_prompt_text,
            embedding_model=self._model.embedding_model_name,
        )

    def retrain_from_file(self, path: str | Path) -> None:
        """Load additional examples from a file and retrain in-place."""
        from nlq_validator.loader import FileLoader

        self.retrain(FileLoader.load(path))

    def apply_calibration(self, result: CalibrationResult) -> None:
        """Apply the suggested threshold from a CalibrationResult."""
        self._threshold = result.suggested_threshold

    def calibrate(
        self,
        in_domain: list[str],
        off_domain: list[str],
    ) -> CalibrationResult:
        """Score sample queries and find the F1-optimal threshold."""
        in_scores = [self._model.score(q) for q in in_domain]
        off_scores = [self._model.score(q) for q in off_domain]

        all_scores = in_scores + off_scores
        labels = [1] * len(in_scores) + [0] * len(off_scores)
        candidates = sorted(set(all_scores))

        best_f1 = -1.0
        best_t = candidates[0] if candidates else 0.25

        for t in candidates:
            preds = [1 if s >= t else 0 for s in all_scores]
            tp = sum(p == 1 and lb == 1 for p, lb in zip(preds, labels))
            fp = sum(p == 1 and lb == 0 for p, lb in zip(preds, labels))
            fn = sum(p == 0 and lb == 1 for p, lb in zip(preds, labels))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        return CalibrationResult(
            in_domain_scores=in_scores,
            off_domain_scores=off_scores,
            suggested_threshold=best_t,
        )

    def validate(self, query: str) -> ValidationResult:
        s = self._model.score(query)
        if s < self._threshold:
            return ValidationResult(
                is_valid=False,
                errors=[f"Query appears off-topic (score={s:.3f}, threshold={self._threshold:.3f})"],
            )
        return ValidationResult(is_valid=True)

    def score(self, query: str) -> float:
        return self._model.score(query)

    @property
    def threshold(self) -> float:
        return self._threshold
