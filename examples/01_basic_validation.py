"""
01 — Basic validation
=====================
Train a validator from a questions file + system prompt, then validate queries.
Shows the core workflow: train → validate → save → load → validate again.
"""
from pathlib import Path

from nlq_validator import NLQValidator

HERE = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a SQL assistant. You help users write queries, "
    "understand JOINs, indexes, and query optimization."
)

# ── Train ─────────────────────────────────────────────────────────────────────
print("Training from file...")
try:
    v = NLQValidator.from_training_file(HERE / "questions.txt", SYSTEM_PROMPT)
    print("Model trained successfully.\n")
except FileNotFoundError as e:
    raise SystemExit(f"Training file not found: {e}")
except ValueError as e:
    raise SystemExit(f"Invalid training data: {e}")

# ── Validate ──────────────────────────────────────────────────────────────────
queries = [
    ("How do I optimize a slow JOIN?",   True),
    ("What is a primary key?",           True),
    ("What is inner join?",              True),
    ("What is a foreign key?",           True),
    ("What is my horoscope today?",      False),
    ("How do I bake sourdough bread?",   False),
]

print(f"{'Query':<45} {'Expected':<10} {'Result':<6} {'Score'}")
print("-" * 75)
for query, expected in queries:
    result = v.validate(query)
    status = "PASS" if result.is_valid else "FAIL"
    match  = "OK" if result.is_valid == expected else "!!"
    print(f"{query:<45} {'in-domain' if expected else 'off-topic':<10} [{status}] {match}  {v.score(query):.3f}")

# ── Save / Load round-trip ─────────────────────────────────────────────────────
model_path = HERE / "saved_model.pkl"
v.save(model_path)
print(f"\nModel saved to {model_path.name}")

v2 = NLQValidator.load(model_path)
print(f"Model loaded. Score for 'What is a SQL JOIN?': {v2.score('What is a SQL JOIN?'):.3f}")
