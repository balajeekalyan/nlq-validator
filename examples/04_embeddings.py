"""
04 — Semantic embeddings (sentence-transformers)
================================================
TF-IDF matches exact words. Embeddings match *meaning*.

Example: "How do I speed up a slow query?" and "What are performance tips for SQL?"
share almost no words but are semantically identical. TF-IDF misses this;
sentence-transformers catches it.

Install the extra before running:
    pip install 'nlq-validator[embeddings]'

Popular embedding models (all free, run locally):
    all-MiniLM-L6-v2     — fast, 80 MB, good general purpose
    all-mpnet-base-v2    — more accurate, 420 MB
    paraphrase-MiniLM-L3-v2 — very fast, 61 MB
"""
from pathlib import Path

from nlq_validator import NLQValidator

HERE = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a SQL assistant. You help users write queries, "
    "understand JOINs, indexes, and query optimization."
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Train without embeddings (TF-IDF only) ────────────────────────────────────
print("Training without embeddings (TF-IDF only)...")
v_tfidf = NLQValidator.from_training_file(HERE / "questions.txt", SYSTEM_PROMPT)
print("Done.\n")

# ── Train with embeddings ─────────────────────────────────────────────────────
print(f"Training with embeddings ({EMBEDDING_MODEL})...")
print("(First run downloads the model — ~80 MB, cached afterwards)")
try:
    v_emb = NLQValidator.from_training_file(
        HERE / "questions.txt",
        SYSTEM_PROMPT,
        embedding_model=EMBEDDING_MODEL,
    )
    print("Done.\n")
except ImportError:
    raise SystemExit(
        "sentence-transformers is not installed.\n"
        "Run: pip install 'nlq-validator[embeddings]'"
    )

# ── Compare scores on paraphrased queries ─────────────────────────────────────
# These queries are semantically in-domain but use different words than the
# training examples — TF-IDF will score them low, embeddings will score them high.
paraphrased = [
    "How do I speed up a slow database query?",        # ≈ "optimize query"
    "What are performance tips for SQL?",              # ≈ "query optimization"
    "Explain the difference between table relations",  # ≈ "JOINs / foreign keys"
]

print(f"{'Query':<50} {'TF-IDF':>8}  {'Embeddings':>10}")
print("-" * 72)
for q in paraphrased:
    s_tfidf = v_tfidf.score(q)
    s_emb   = v_emb.score(q)
    gain    = f"(+{s_emb - s_tfidf:.3f})" if s_emb > s_tfidf else ""
    print(f"{q:<50} {s_tfidf:>8.3f}  {s_emb:>10.3f}  {gain}")

# ── Off-topic queries should still score low with embeddings ──────────────────
print()
off_topic = [
    "How do I bake sourdough bread?",
    "What is my horoscope today?",
]
print(f"{'Off-topic query':<50} {'TF-IDF':>8}  {'Embeddings':>10}")
print("-" * 72)
for q in off_topic:
    print(f"{q:<50} {v_tfidf.score(q):>8.3f}  {v_emb.score(q):>10.3f}")

# ── Save and reload (encoder is re-instantiated lazily after load) ─────────────
model_path = HERE / "saved_model_with_embeddings.pkl"
v_emb.save(model_path)
print(f"\nSaved to {model_path.name}")

v_loaded = NLQValidator.load(model_path)
score_after_load = v_loaded.score("How do I speed up a slow database query?")
print(f"Score after load: {score_after_load:.3f}  (encoder reloaded lazily on first score call)")
model_path.unlink()
