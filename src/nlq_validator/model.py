from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


@dataclass
class TopicModel:
    vectorizer: TfidfVectorizer
    train_matrix: csr_matrix
    centroid: np.ndarray
    training_examples: list[str]
    system_prompt_vec: np.ndarray | None = field(default=None)

    # Retrain bookkeeping — user examples only, no system prompt sentences
    user_examples: list[str] = field(default_factory=list)
    system_prompt_text: str | None = field(default=None)

    # Optional semantic embedding support (sentence-transformers)
    embedding_model_name: str | None = field(default=None)
    example_embeddings: np.ndarray | None = field(default=None)
    centroid_embedding: np.ndarray | None = field(default=None)

    # Live encoder — never serialized, re-instantiated lazily after load
    _encoder: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._encoder = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_encoder", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._encoder = None

    def _get_encoder(self):
        if self.embedding_model_name is None:
            return None
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return None
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def score(self, query: str) -> float:
        vec = self.vectorizer.transform([query])
        centroid_sim = float(cosine_similarity(vec, self.centroid)[0][0])
        nn_sim = float(cosine_similarity(vec, self.train_matrix).max())
        candidates = [centroid_sim, nn_sim]
        if self.system_prompt_vec is not None:
            sp_sim = float(cosine_similarity(vec, self.system_prompt_vec)[0][0])
            candidates.append(sp_sim)

        encoder = self._get_encoder()
        if (
            encoder is not None
            and self.example_embeddings is not None
            and self.centroid_embedding is not None
        ):
            q_emb = encoder.encode([query], normalize_embeddings=True)
            emb_nn_sim = float(cosine_similarity(q_emb, self.example_embeddings).max())
            emb_centroid_sim = float(cosine_similarity(q_emb, self.centroid_embedding)[0][0])
            candidates.extend([emb_nn_sim, emb_centroid_sim])

        return max(candidates)
