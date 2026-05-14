# nlq-validator

A lightweight Natural Language Query (NLQ) validator that keeps your LLM assistant on-topic. Train it on a handful of example questions, and it will accept in-domain queries while rejecting off-topic ones — no server, no API key required for the core functionality.

## Features

- **TF-IDF scoring** out of the box — no model downloads needed
- **Semantic embeddings** via sentence-transformers for paraphrase-aware matching
- **Threshold calibration** — find the F1-optimal cutoff for your domain
- **Incremental retraining** — add examples without rebuilding from scratch
- **LLM-powered question generation** — auto-generate training data from your system prompt (Claude, ChatGPT, Gemini, Mistral, Grok, Perplexity)
- **Async support** for all LLM integrations
- **Zero runtime dependencies** beyond scikit-learn for the core validator

## Installation

```bash
pip install nlq-validator
```

### Optional extras

```bash
pip install 'nlq-validator[embeddings]'   # sentence-transformers for semantic matching
pip install 'nlq-validator[anthropic]'    # Claude integration
pip install 'nlq-validator[openai]'       # ChatGPT, Grok, Perplexity integrations
pip install 'nlq-validator[gemini]'       # Google Gemini integration
pip install 'nlq-validator[mistral]'      # Mistral integration
pip install 'nlq-validator[all-llm]'      # All LLM integrations
```

## Quick start

```python
from nlq_validator import NLQValidator

SYSTEM_PROMPT = (
    "You are a SQL assistant. You help users write queries, "
    "understand JOINs, indexes, and query optimization."
)

# Train from a plain-text file (one question per line)
v = NLQValidator.from_training_file("questions.txt", SYSTEM_PROMPT)

result = v.validate("How do I write a SELECT statement?")
print(result.is_valid)   # True

result = v.validate("What is my horoscope today?")
print(result.is_valid)   # False
print(result.errors)     # ['Query appears off-topic (score=0.000, threshold=0.250)']
```

## Training data format

Supported file formats: `.txt` (one question per line), `.csv` (first column), `.json` (list of strings or list of `{"text": "..."}` objects).

```
# questions.txt
How do I write a SELECT statement?
What is a SQL JOIN?
How do I filter rows with WHERE clause?
What is the difference between INNER JOIN and LEFT JOIN?
...
```

## Threshold calibration

The default threshold of `0.25` is a conservative starting point. Use `calibrate()` to find the optimal value for your domain:

```python
in_domain = ["How do I use GROUP BY?", "What is a primary key?", ...]
off_domain = ["How do I bake bread?", "What is my horoscope?", ...]

result = v.calibrate(in_domain, off_domain)
result.summary()          # prints precision/recall/F1 table
v.apply_calibration(result)  # applies suggested threshold
```

## Incremental retraining

```python
v.retrain(["How do I write a CTE?", "What is a window function?"])
# or from a file:
v.retrain_from_file("more_questions.txt")
```

## Semantic embeddings

For queries that use different words but mean the same thing:

```python
v = NLQValidator.from_training_file(
    "questions.txt",
    SYSTEM_PROMPT,
    embedding_model="all-MiniLM-L6-v2",   # requires nlq-validator[embeddings]
)
```

## LLM-powered question generation

Generate training data automatically from your system prompt:

```python
from nlq_validator.integrations.claude import ClaudeIntegration

llm = ClaudeIntegration()   # reads ANTHROPIC_API_KEY env var
v = NLQValidator.from_llm(llm, SYSTEM_PROMPT, count=50)
```

Async variant:

```python
v = await NLQValidator.from_llm_async(llm, SYSTEM_PROMPT, count=50)
```

### Supported LLM providers

| Provider    | Extra              | Class                 | Env vars                              |
|-------------|--------------------|-----------------------|---------------------------------------|
| Claude      | `[anthropic]`      | `ClaudeIntegration`   | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`|
| ChatGPT     | `[openai]`         | `ChatGPTIntegration`  | `OPENAI_API_KEY`, `OPENAI_MODEL`      |
| Gemini      | `[gemini]`         | `GeminiIntegration`   | `GEMINI_API_KEY`, `GEMINI_MODEL`      |
| Mistral     | `[mistral]`        | `MistralIntegration`  | `MISTRAL_API_KEY`, `MISTRAL_MODEL`    |
| Grok        | `[openai]`         | `GrokIntegration`     | `XAI_API_KEY`, `XAI_MODEL`            |
| Perplexity  | `[openai]`         | `PerplexityIntegration`| `PERPLEXITY_API_KEY`, `PERPLEXITY_MODEL`|

## Save and load

```python
v.save("my_model.pkl")
v2 = NLQValidator.load("my_model.pkl")
```

## License

MIT — see [LICENSE](LICENSE).
