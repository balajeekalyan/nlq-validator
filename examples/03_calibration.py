"""
03 — Threshold calibration
==========================
The default threshold (0.25) is a reasonable starting point, but your domain
may need a tighter or looser value. This example shows how to use calibrate()
to find the F1-optimal threshold for your specific data.

Workflow:
  1. Collect a small labelled sample (in-domain + off-domain queries)
  2. Call calibrate() to see precision/recall/F1 at every possible threshold
  3. Apply the suggested threshold (or pick your own from the table)
"""
from pathlib import Path

from nlq_validator import NLQValidator

HERE = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a SQL assistant. You help users write queries, "
    "understand JOINs, indexes, and query optimization."
)

v = NLQValidator.from_training_file(HERE / "questions.txt", SYSTEM_PROMPT)

# ── Sample queries for calibration ────────────────────────────────────────────
in_domain = [
    "How do I write a SELECT statement?",
    "What is a SQL JOIN?",
    "How do I use GROUP BY?",
    "What is a primary key?",
    "How do I create an index?",
    "What is a foreign key constraint?",
    "How do I filter rows with WHERE?",
    "What does DISTINCT do?",
]

off_domain = [
    "How do I bake sourdough bread?",
    "What is my horoscope today?",
    "How do I fix my car engine?",
    "What is the best smartphone to buy?",
    "Tell me a joke.",
    "How do I learn to play guitar?",
    "What is the capital of France?",
    "How do I lose weight fast?",
]

# ── Run calibration ────────────────────────────────────────────────────────────
print("Running calibration...\n")
result = v.calibrate(in_domain, off_domain)

print("Score distributions:")
print(f"  In-domain  — min: {min(result.in_domain_scores):.3f}  "
      f"max: {max(result.in_domain_scores):.3f}  "
      f"avg: {sum(result.in_domain_scores)/len(result.in_domain_scores):.3f}")
print(f"  Off-domain — min: {min(result.off_domain_scores):.3f}  "
      f"max: {max(result.off_domain_scores):.3f}  "
      f"avg: {sum(result.off_domain_scores)/len(result.off_domain_scores):.3f}")

print()
result.summary()

# ── Apply the suggested threshold ─────────────────────────────────────────────
print(f"\nCurrent threshold : {v.threshold:.4f}")
print(f"Suggested threshold: {result.suggested_threshold:.4f}")

v.apply_calibration(result)
print(f"Threshold updated to {v.threshold:.4f}\n")

# ── Re-validate with new threshold ────────────────────────────────────────────
test_queries = [
    ("What is a SQL JOIN?",           "in-domain"),
    ("How do I optimize a slow JOIN?","in-domain"),
    ("How do I bake bread?",          "off-topic"),
    ("What is my horoscope?",         "off-topic"),
]

print("Validation with updated threshold:")
for query, label in test_queries:
    r = v.validate(query)
    print(f"  [{'PASS' if r.is_valid else 'FAIL'}] ({label})  {query}")
