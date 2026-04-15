"""
05 — Generate training questions with an LLM
============================================
Instead of writing questions by hand, use any supported LLM to generate them
automatically from your system prompt.

Supported integrations (install the matching extra):
    pip install 'nlq-validator[openai]'      # ChatGPT, Grok, Perplexity
    pip install 'nlq-validator[anthropic]'   # Claude
    pip install 'nlq-validator[gemini]'      # Gemini
    pip install 'nlq-validator[mistral]'     # Mistral

Set the corresponding API key as an environment variable, e.g.:
    set ANTHROPIC_API_KEY=sk-ant-...
    set ANTHROPIC_MODEL=claude-haiku-4-5-20251001

Then run this file.  By default it uses Claude — change INTEGRATION below to
swap providers without touching anything else.
"""
import os
from pathlib import Path

from nlq_validator import NLQValidator

HERE = Path(__file__).parent

SYSTEM_PROMPT = (
    "You are a SQL assistant. You help users write queries, "
    "understand JOINs, indexes, and query optimization."
)

# ── Pick your integration ──────────────────────────────────────────────────────
# Uncomment the one you want to use.

from nlq_validator.integrations.claude import ClaudeIntegration
llm = ClaudeIntegration()                    # reads ANTHROPIC_API_KEY + ANTHROPIC_MODEL

# from nlq_validator.integrations.chatgpt import ChatGPTIntegration
# llm = ChatGPTIntegration()                 # reads OPENAI_API_KEY + OPENAI_MODEL

# from nlq_validator.integrations.gemini import GeminiIntegration
# llm = GeminiIntegration()                  # reads GEMINI_API_KEY + GEMINI_MODEL

# from nlq_validator.integrations.mistral import MistralIntegration
# llm = MistralIntegration()                 # reads MISTRAL_API_KEY + MISTRAL_MODEL

# from nlq_validator.integrations.grok import GrokIntegration
# llm = GrokIntegration()                    # reads XAI_API_KEY + XAI_MODEL

# from nlq_validator.integrations.perplexity import PerplexityIntegration
# llm = PerplexityIntegration()              # reads PERPLEXITY_API_KEY + PERPLEXITY_MODEL

# ── Option A: generate, inspect, save to file, then train ─────────────────────
# Useful when you want to review or edit the questions before training.

questions_file = HERE / "generated_questions.txt"

print(f"Generating 20 questions using {llm.__class__.__name__}...")
try:
    questions = llm.generate_and_save(SYSTEM_PROMPT, questions_file, count=20)
except Exception as e:
    raise SystemExit(f"LLM call failed: {e}")

print(f"Generated {len(questions)} questions → saved to {questions_file.name}\n")
print("Sample questions:")
for q in questions[:5]:
    print(f"  • {q}")
print("  ...")

v = NLQValidator.from_training_file(questions_file, SYSTEM_PROMPT)
print(f"\nModel trained on {len(questions)} generated questions.")

# ── Option B: generate and train in one step ──────────────────────────────────
# Useful when you trust the LLM output and don't need to inspect the questions.

# v = NLQValidator.from_llm(llm, SYSTEM_PROMPT, count=20)

# ── Validate ──────────────────────────────────────────────────────────────────
test_queries = [
    ("How do I write a SELECT statement?",  True),
    ("What is a SQL JOIN?",                 True),
    ("How do I optimize a slow query?",     True),
    ("How do I bake sourdough bread?",      False),
    ("What is my horoscope today?",         False),
]

print()
print(f"{'Query':<45} {'Expected':<10} {'Result'}")
print("-" * 70)
for query, expected in test_queries:
    result = v.validate(query)
    status = "PASS" if result.is_valid else "FAIL"
    match  = "OK" if result.is_valid == expected else "!!"
    print(f"{query:<45} {'in-domain' if expected else 'off-topic':<10} [{status}] {match}  score={v.score(query):.3f}")
