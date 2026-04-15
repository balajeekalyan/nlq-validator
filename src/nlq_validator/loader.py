import csv
import json
from pathlib import Path


class FileLoader:
    _JSON_TEXT_KEYS = ("text", "query", "question")

    @staticmethod
    def load(path: str | Path) -> list[str]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".txt":
            examples = FileLoader._load_txt(path)
        elif suffix == ".csv":
            examples = FileLoader._load_csv(path)
        elif suffix == ".json":
            examples = FileLoader._load_json(path)
        else:
            raise ValueError(f"Unsupported file extension '{suffix}'. Use .txt, .csv, or .json.")
        if len(examples) < 2:
            raise ValueError(f"Training file must contain at least 2 examples, got {len(examples)}.")
        return examples

    @staticmethod
    def _load_txt(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _load_csv(path: Path) -> list[str]:
        examples: list[str] = []
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return examples
        # Skip header if first cell doesn't look like a sample query
        start = 1 if rows and rows[0] and not rows[0][0].strip().endswith("?") and len(rows) > 1 and rows[0][0].strip().replace(" ", "").isalpha() else 0
        for row in rows[start:]:
            if row and row[0].strip():
                examples.append(row[0].strip())
        return examples

    @staticmethod
    def _load_json(path: Path) -> list[str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a top-level list.")
        examples: list[str] = []
        for item in data:
            if isinstance(item, str):
                if item.strip():
                    examples.append(item.strip())
            elif isinstance(item, dict):
                for key in FileLoader._JSON_TEXT_KEYS:
                    if key in item and isinstance(item[key], str) and item[key].strip():
                        examples.append(item[key].strip())
                        break
            else:
                raise ValueError(f"JSON list items must be strings or objects, got {type(item).__name__}.")
        return examples
