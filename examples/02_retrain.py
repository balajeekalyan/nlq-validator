"""
02 — Retrain
============
Demonstrates adding new examples to an already-trained model without
rebuilding from scratch.

Use case: your domain grows over time (e.g. you add support for CTEs and
window functions) and you want the validator to accept those queries without
discarding the original training data.
"""
from pathlib import Path

from nlq_validator import NLQValidator

HERE = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a SQL assistant. You help users write queries, "
    "understand JOINs, indexes, and query optimization."
)

v = NLQValidator.from_training_file(HERE / "questions.txt", SYSTEM_PROMPT)

# ── Before retrain ─────────────────────────────────────────────────────────────
new_queries = [
    "How do I write a CTE?",
    "What is a window function in SQL?",
    "How do I use RANK() OVER PARTITION BY?",
]

print("Before retrain:")
for q in new_queries:
    r = v.validate(q)
    print(f"  [{'PASS' if r.is_valid else 'FAIL'}] (score={v.score(q):.3f})  {q}")

# ── Retrain with new examples ──────────────────────────────────────────────────
additional = [
    "How do I write a CTE in SQL?",
    "What is a Common Table Expression?",
    "How do I use window functions like ROW_NUMBER?",
    "What does PARTITION BY do in a window function?",
    "How do I use RANK() and DENSE_RANK()?",
]

print(f"\nRetraining with {len(additional)} new examples...")
v.retrain(additional)
print("Retrain complete.\n")

# ── After retrain ──────────────────────────────────────────────────────────────
print("After retrain:")
for q in new_queries:
    r = v.validate(q)
    print(f"  [{'PASS' if r.is_valid else 'FAIL'}] (score={v.score(q):.3f})  {q}")

# ── Original examples still work ───────────────────────────────────────────────
print("\nOriginal examples still accepted:")
original = ["What is a primary key?", "What is inner join?", "What is a foreign key?"]
for q in original:
    r = v.validate(q)
    print(f"  [{'PASS' if r.is_valid else 'FAIL'}] (score={v.score(q):.3f})  {q}")

# ── Retrain from a file ────────────────────────────────────────────────────────
extra_file = HERE / "extra_questions.txt"
extra_file.write_text(
    "How do I use LEAD and LAG functions?\nWhat is a recursive CTE?\n",
    encoding="utf-8",
)
v.retrain_from_file(extra_file)
print(f"\nRetrained from {extra_file.name}.")
print(f"  Score for 'How do I use LAG()': {v.score('How do I use LAG()?'):.3f}")
extra_file.unlink()
