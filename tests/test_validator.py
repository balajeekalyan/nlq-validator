import csv
import json
from pathlib import Path

import pytest

from nlq_validator import NLQValidator, ValidationResult
from nlq_validator.loader import FileLoader
from nlq_validator.model import TopicModel
from nlq_validator.persistence import load_model, save_model
from nlq_validator.trainer import train

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SQL_SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. You help users write SQL queries, "
    "understand database concepts, and troubleshoot query errors. "
    "You answer questions about SELECT, INSERT, UPDATE, DELETE, JOINs, "
    "indexes, primary keys, foreign keys, and query optimization."
)

SQL_EXAMPLES = [
    "How do I write a SELECT statement?",
    "What is a SQL JOIN?",
    "How do I filter rows with WHERE clause?",
    "What is the difference between INNER JOIN and LEFT JOIN?",
    "How do I group results using GROUP BY?",
    "What does DISTINCT do in SQL?",
    "How do I count rows in a table?",
    "What is a primary key?",
    "How do I insert data into a table?",
    "How do I update existing records?",
    "What is a foreign key constraint?",
    "How do I delete rows from a table?",
    "What is a database index?",
    "How do I create a new table?",
    "What is the difference between UNION and UNION ALL?",
]


@pytest.fixture
def tmp_txt_file(tmp_path: Path) -> Path:
    f = tmp_path / "questions.txt"
    f.write_text("\n".join(SQL_EXAMPLES), encoding="utf-8")
    return f


@pytest.fixture
def trained_validator(tmp_txt_file: Path) -> NLQValidator:
    return NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# TestFileLoader
# ---------------------------------------------------------------------------


class TestFileLoader:
    def test_load_txt_strips_blanks(self, tmp_path: Path) -> None:
        f = tmp_path / "q.txt"
        f.write_text("line one\n\n  \nline two\n", encoding="utf-8")
        result = FileLoader.load(f)
        assert result == ["line one", "line two"]

    def test_load_csv_first_column(self, tmp_path: Path) -> None:
        f = tmp_path / "q.csv"
        with f.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["question", "category"])
            writer.writerow(["What is SQL?", "db"])
            writer.writerow(["How do I JOIN tables?", "db"])
        result = FileLoader.load(f)
        assert "What is SQL?" in result
        assert "How do I JOIN tables?" in result
        assert "category" not in result

    def test_load_json_list_of_strings(self, tmp_path: Path) -> None:
        f = tmp_path / "q.json"
        f.write_text(json.dumps(SQL_EXAMPLES[:5]), encoding="utf-8")
        result = FileLoader.load(f)
        assert result == SQL_EXAMPLES[:5]

    def test_load_json_list_of_objects(self, tmp_path: Path) -> None:
        data = [{"text": q} for q in SQL_EXAMPLES[:5]]
        f = tmp_path / "q.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        result = FileLoader.load(f)
        assert result == SQL_EXAMPLES[:5]

    def test_load_unsupported_extension_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "q.xml"
        f.write_text("<root/>", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            FileLoader.load(f)

    def test_load_too_few_examples_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "q.txt"
        f.write_text("only one line", encoding="utf-8")
        with pytest.raises(ValueError, match="at least 2 examples"):
            FileLoader.load(f)


# ---------------------------------------------------------------------------
# TestTopicModel
# ---------------------------------------------------------------------------


class TestTopicModel:
    def test_score_returns_float_in_range(self) -> None:
        model = train(SQL_EXAMPLES)
        s = model.score("What is a SQL query?")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_in_domain_scores_higher_than_off_domain(self) -> None:
        model = train(SQL_EXAMPLES)
        in_domain = model.score("How do I use GROUP BY in SQL?")
        off_domain = model.score("What is the best recipe for chocolate cake?")
        assert in_domain > off_domain


# ---------------------------------------------------------------------------
# TestTrainer
# ---------------------------------------------------------------------------


class TestTrainer:
    def test_train_returns_topic_model(self) -> None:
        model = train(SQL_EXAMPLES)
        assert isinstance(model, TopicModel)

    def test_train_matrix_shape(self) -> None:
        model = train(SQL_EXAMPLES)
        assert model.train_matrix.shape[0] == len(SQL_EXAMPLES)


# ---------------------------------------------------------------------------
# TestPersistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        model = train(SQL_EXAMPLES)
        path = tmp_path / "model.pkl"
        save_model(model, path)
        loaded = load_model(path)
        assert model.score("SELECT query") == loaded.score("SELECT query")

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")


# ---------------------------------------------------------------------------
# TestNLQValidator
# ---------------------------------------------------------------------------


class TestNLQValidator:
    def test_from_training_file_validates_in_domain(self, trained_validator: NLQValidator) -> None:
        result = trained_validator.validate("How do I write a SELECT statement in SQL?")
        assert result.is_valid is True

    def test_from_training_file_rejects_off_domain(self, trained_validator: NLQValidator) -> None:
        result = trained_validator.validate("How do I bake sourdough bread?")
        assert result.is_valid is False
        assert result.errors

    def test_save_and_load_classmethod(self, trained_validator: NLQValidator, tmp_path: Path) -> None:
        path = tmp_path / "model.pkl"
        trained_validator.save(path)
        loaded = NLQValidator.load(path)
        result = loaded.validate("What is a SQL JOIN?")
        assert result.is_valid is True

    def test_validate_error_message_contains_score(self, trained_validator: NLQValidator) -> None:
        result = trained_validator.validate("What is my horoscope today?")
        assert result.is_valid is False
        assert any("score=" in e for e in result.errors)

    def test_score_method_exposed(self, trained_validator: NLQValidator) -> None:
        s = trained_validator.score("SELECT * FROM table")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_custom_threshold_high_rejects_more(self, tmp_txt_file: Path) -> None:
        validator = NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT, threshold=0.99)
        result = validator.validate("How do I write a SELECT statement in SQL?")
        assert result.is_valid is False

    def test_custom_threshold_zero_accepts_all(self, tmp_txt_file: Path) -> None:
        validator = NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT, threshold=0.0)
        result = validator.validate("What is my horoscope today?")
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# TestSystemPrompt
# ---------------------------------------------------------------------------

COOKING_SYSTEM_PROMPT = (
    "You are a professional chef assistant. You help users with recipes, "
    "cooking techniques, ingredient substitutions, and meal planning. "
    "You answer questions about baking, grilling, seasoning, and kitchen tools."
)


class TestSystemPrompt:
    def test_from_training_file_with_system_prompt_covers_prompt_vocabulary(
        self, tmp_txt_file: Path
    ) -> None:
        with_prompt = NLQValidator.from_training_file(
            tmp_txt_file, system_prompt=SQL_SYSTEM_PROMPT
        )
        result = with_prompt.validate("How do I optimize SQL JOINs and indexes?")
        assert result.is_valid is True

    def test_from_training_file_with_system_prompt_rejects_off_topic(
        self, tmp_txt_file: Path
    ) -> None:
        validator = NLQValidator.from_training_file(
            tmp_txt_file, system_prompt=SQL_SYSTEM_PROMPT
        )
        result = validator.validate("What is the best way to roast a chicken?")
        assert result.is_valid is False

    def test_system_prompt_expands_coverage_beyond_file(
        self, tmp_txt_file: Path
    ) -> None:
        # "query optimization" and "troubleshoot" are in the system prompt but not
        # in the training file. The combined model should still accept such queries.
        with_prompt = NLQValidator.from_training_file(
            tmp_txt_file, system_prompt=SQL_SYSTEM_PROMPT
        )
        result = with_prompt.validate("How do I troubleshoot slow query errors?")
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# TestRetrain
# ---------------------------------------------------------------------------


class TestRetrain:
    def test_retrain_adds_new_vocabulary(self, tmp_txt_file: Path) -> None:
        validator = NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT)
        new_examples = [
            "How do I use window functions?",
            "What is a CTE in SQL?",
            "How do I write a recursive query?",
        ]
        validator.retrain(new_examples)
        result = validator.validate("How do I write a CTE query?")
        assert result.is_valid is True

    def test_retrain_preserves_original_examples(self, trained_validator: NLQValidator) -> None:
        trained_validator.retrain(["How do I use window functions?"])
        result = trained_validator.validate("What is a primary key?")
        assert result.is_valid is True

    def test_retrain_does_not_duplicate_system_prompt_sentences(
        self, tmp_txt_file: Path
    ) -> None:
        validator = NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT)
        n_before = validator._model.train_matrix.shape[0]
        validator.retrain(["How do I use window functions?"])
        n_after = validator._model.train_matrix.shape[0]
        # Should grow by 1 (new example) + system prompt sentences (same count), not double
        assert n_after == n_before + 1

    def test_retrain_from_file(self, tmp_txt_file: Path, tmp_path: Path) -> None:
        validator = NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT)
        extra_file = tmp_path / "extra.txt"
        extra_file.write_text("How do I use window functions?\nWhat is a CTE?", encoding="utf-8")
        validator.retrain_from_file(extra_file)
        assert validator.validate("How do I write a CTE query?").is_valid is True

    def test_double_retrain_is_stable(self, tmp_txt_file: Path) -> None:
        validator = NLQValidator.from_training_file(tmp_txt_file, SQL_SYSTEM_PROMPT)
        validator.retrain(["How do I use window functions?"])
        validator.retrain(["What is a CTE?"])
        assert validator.validate("What is a primary key?").is_valid is True


# ---------------------------------------------------------------------------
# TestCalibration
# ---------------------------------------------------------------------------

IN_DOMAIN_CALIBRATION = [
    "How do I write a SELECT statement?",
    "What is a SQL JOIN?",
    "How do I use GROUP BY?",
    "What is a primary key?",
    "How do I create an index?",
]
OFF_DOMAIN_CALIBRATION = [
    "How do I bake bread?",
    "What is my horoscope today?",
    "How do I fix my car?",
    "What is the weather like?",
    "Tell me a joke.",
]


class TestCalibration:
    def test_calibrate_returns_calibration_result(
        self, trained_validator: NLQValidator
    ) -> None:
        from nlq_validator import CalibrationResult

        result = trained_validator.calibrate(IN_DOMAIN_CALIBRATION, OFF_DOMAIN_CALIBRATION)
        assert isinstance(result, CalibrationResult)

    def test_calibrate_score_counts_match_input(
        self, trained_validator: NLQValidator
    ) -> None:
        result = trained_validator.calibrate(IN_DOMAIN_CALIBRATION, OFF_DOMAIN_CALIBRATION)
        assert len(result.in_domain_scores) == len(IN_DOMAIN_CALIBRATION)
        assert len(result.off_domain_scores) == len(OFF_DOMAIN_CALIBRATION)

    def test_calibrate_in_domain_scores_higher(
        self, trained_validator: NLQValidator
    ) -> None:
        result = trained_validator.calibrate(IN_DOMAIN_CALIBRATION, OFF_DOMAIN_CALIBRATION)
        assert sum(result.in_domain_scores) > sum(result.off_domain_scores)

    def test_calibrate_suggested_threshold_is_float(
        self, trained_validator: NLQValidator
    ) -> None:
        result = trained_validator.calibrate(IN_DOMAIN_CALIBRATION, OFF_DOMAIN_CALIBRATION)
        assert isinstance(result.suggested_threshold, float)

    def test_calibrate_summary_runs_without_error(
        self, trained_validator: NLQValidator, capsys
    ) -> None:
        result = trained_validator.calibrate(IN_DOMAIN_CALIBRATION, OFF_DOMAIN_CALIBRATION)
        result.summary()
        captured = capsys.readouterr()
        assert "Threshold" in captured.out
        assert "Suggested threshold" in captured.out


# ---------------------------------------------------------------------------
# TestApplyCalibration
# ---------------------------------------------------------------------------


class TestApplyCalibration:
    def test_apply_calibration_updates_threshold(self, trained_validator: NLQValidator) -> None:
        from nlq_validator import CalibrationResult

        result = CalibrationResult(
            in_domain_scores=[0.8, 0.9],
            off_domain_scores=[0.1, 0.05],
            suggested_threshold=0.45,
        )
        trained_validator.apply_calibration(result)
        assert trained_validator.threshold == 0.45

    def test_apply_calibration_affects_validation(self, trained_validator: NLQValidator) -> None:
        # "How do I optimize a slow JOIN?" scores ~0.6 — passes at 0.25 but fails at 0.99.
        from nlq_validator import CalibrationResult

        query = "How do I optimize a slow JOIN?"
        assert trained_validator.validate(query).is_valid is True

        result = CalibrationResult([], [], suggested_threshold=0.99)
        trained_validator.apply_calibration(result)
        assert trained_validator.validate(query).is_valid is False


# ---------------------------------------------------------------------------
# TestAsync
# ---------------------------------------------------------------------------


class TestAsync:
    def test_generate_and_save_async_signature(self) -> None:
        import inspect
        from nlq_validator.integrations.base import BaseLLMIntegration
        assert inspect.iscoroutinefunction(BaseLLMIntegration.generate_and_save_async)

    def test_generate_questions_async_is_abstract(self) -> None:
        import inspect
        from nlq_validator.integrations.base import BaseLLMIntegration
        assert inspect.iscoroutinefunction(BaseLLMIntegration.generate_questions_async)

    def test_all_integrations_implement_async(self) -> None:
        import inspect
        from nlq_validator.integrations.claude import ClaudeIntegration
        from nlq_validator.integrations.chatgpt import ChatGPTIntegration
        from nlq_validator.integrations.gemini import GeminiIntegration
        from nlq_validator.integrations.mistral import MistralIntegration
        from nlq_validator.integrations.grok import GrokIntegration
        from nlq_validator.integrations.perplexity import PerplexityIntegration

        for cls in [ClaudeIntegration, ChatGPTIntegration, GeminiIntegration,
                    MistralIntegration, GrokIntegration, PerplexityIntegration]:
            assert inspect.iscoroutinefunction(cls.generate_questions_async), \
                f"{cls.__name__} missing async implementation"

    def test_from_llm_async_is_coroutine(self) -> None:
        import inspect
        assert inspect.iscoroutinefunction(NLQValidator.from_llm_async)
