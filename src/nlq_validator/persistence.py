from pathlib import Path

import joblib

from nlq_validator.model import TopicModel


def save_model(model: TopicModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def load_model(path: str | Path) -> TopicModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
