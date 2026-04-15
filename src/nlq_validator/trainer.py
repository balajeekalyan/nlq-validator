import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from nlq_validator.model import TopicModel


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences on punctuation boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def train(
    examples: list[str],
    system_prompt: str | None = None,
    embedding_model: str | None = None,
) -> TopicModel:
    user_examples = list(examples)

    corpus = list(examples)
    if system_prompt:
        corpus.extend(_split_sentences(system_prompt))

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english"
    )
    train_matrix = vectorizer.fit_transform(corpus)
    centroid = np.asarray(train_matrix.mean(axis=0))

    system_prompt_vec: np.ndarray | None = None
    if system_prompt:
        sp_sparse = vectorizer.transform([system_prompt])
        system_prompt_vec = np.asarray(sp_sparse.todense())

    example_embeddings: np.ndarray | None = None
    centroid_embedding: np.ndarray | None = None
    if embedding_model is not None:
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(embedding_model)
            example_embeddings = encoder.encode(corpus, normalize_embeddings=True)
            centroid_embedding = example_embeddings.mean(axis=0, keepdims=True)
        except ImportError:
            pass

    return TopicModel(
        vectorizer=vectorizer,
        train_matrix=train_matrix,
        centroid=centroid,
        training_examples=corpus,
        system_prompt_vec=system_prompt_vec,
        user_examples=user_examples,
        system_prompt_text=system_prompt,
        embedding_model_name=embedding_model,
        example_embeddings=example_embeddings,
        centroid_embedding=centroid_embedding,
    )
