# Vendor Verifier

A multi-model tool call evaluation framework for OpenAI-compatible APIs. Tests whether different vendors correctly handle tool calling for any given model.

## Quick Start

```bash
# Install dependencies
uv sync

# Run evaluation
python -m vendor_verifier ./tool-calls/samples_cleaned_cleaned.jsonl \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key YOUR_API_KEY
```

## Usage

```bash
python -m vendor_verifier <file_path> --model <MODEL> --base-url <URL> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `file_path` | Test set file in JSONL format (one request body per line) |
| `--model` | Model name (e.g. `gpt-4o`, `kimi-k2-0905-preview`, `deepseek-chat`) |
| `--base-url` | API endpoint URL (e.g. `https://api.openai.com/v1`) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--api-key` | `$OPENAI_API_KEY` | API key for authentication |
| `--model-type` | auto-detect | Force a specific model adapter (`generic`, `kimi`) |
| `--temperature` | per-request | Override temperature for all requests |
| `--max-tokens` | per-request | Override max_tokens for all requests |
| `--extra-body` | `{}` | Extra JSON body merged into each request (e.g. `'{"top_p": 0.9}'`) |
| `--concurrency` | `5` | Maximum number of concurrent requests |
| `--output` | `results.jsonl` | Path to save detailed per-request results |
| `--summary` | `summary.json` | Path to save aggregated summary |
| `--timeout` | `600` | Per-request timeout in seconds |
| `--retries` | `3` | Number of retries on failure |
| `--incremental` | off | Only rerun failed requests, preserve successful results |

### Kimi-Specific Arguments

| Argument | Description |
|----------|-------------|
| `--use-raw-completions` | Use `/v1/completions` endpoint (requires `--tokenizer-model`) |
| `--tokenizer-model` | HuggingFace tokenizer name for raw completions mode |

## Examples

### Test any OpenAI-compatible provider

```bash
python -m vendor_verifier samples_cleaned_cleaned.jsonl \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --api-key $OPENAI_API_KEY \
    --concurrency 10 \
    --output gpt4o_results.jsonl \
    --summary gpt4o_summary.json
```

### Test Kimi model via official API

```bash
python -m vendor_verifier samples_cleaned_cleaned.jsonl \
    --model kimi-k2-0905-preview \
    --base-url https://api.moonshot.cn/v1 \
    --api-key $MOONSHOT_API_KEY \
    --temperature 0.6
```

### Test via OpenRouter with a specific provider

```bash
python -m vendor_verifier samples_cleaned_cleaned.jsonl \
    --model moonshotai/kimi-k2-0905 \
    --base-url https://openrouter.ai/api/v1 \
    --api-key $OPENROUTER_API_KEY \
    --model-type kimi \
    --extra-body '{"provider": {"only": ["fireworks"]}}'
```

### Incremental mode (resume after interruption)

```bash
python -m vendor_verifier samples_cleaned_cleaned.jsonl \
    --model gpt-4o \
    --base-url https://api.openai.com/v1 \
    --incremental
```

## Output

### results.jsonl

Each line contains a JSON object with:

```json
{
    "data_index": 1,
    "request": { "...": "..." },
    "response": { "...": "..." },
    "status": "success",
    "finish_reason": "tool_calls",
    "tool_calls_valid": true,
    "duration_ms": 1234,
    "hash": "abc123..."
}
```

### summary.json

Aggregated statistics:

```json
{
    "model": "gpt-4o",
    "success_count": 1950,
    "failure_count": 50,
    "finish_stop": 800,
    "finish_tool_calls": 1150,
    "successful_tool_call_count": 1140,
    "schema_validation_error_count": 10,
    "usage": {
        "prompt_tokens": 500000,
        "completion_tokens": 120000,
        "total_tokens": 620000
    }
}
```

## Compare Results Against Baseline

After running evaluations, use `vendor_verifier.compare` to compare a vendor's results against a baseline (e.g. official API results). This computes two key metrics:

- **ToolCall-Trigger Similarity**: Whether the vendor triggers tool calls at the same times as the baseline (precision / recall / F1).
- **ToolCall-Schema Accuracy**: Of the triggered tool calls, how many have valid JSON schemas.

```bash
python -m vendor_verifier.compare \
    --baseline ./tool-calls/kimi-k2-0905-preview_results.jsonl \
    --vendor ./tool-calls/gpt-4o-results.jsonl \
    --output ./tool-calls/gpt-4o-comparison.json
```

| Argument | Description |
|----------|-------------|
| `--baseline` | Path to baseline results JSONL (e.g. official API output) |
| `--vendor` | Path to vendor results JSONL to evaluate |
| `--output` | Path to save comparison JSON (optional, prints to stdout if omitted) |

### Output format

```json
{
  "total_baseline": 2000,
  "total_vendor": 2000,
  "common_indices": 2000,
  "matched_success": 2000,
  "tool_call_trigger_similarity": {
    "TP": 510,
    "FP": 475,
    "FN": 173,
    "TN": 842,
    "precision": 0.5178,
    "recall": 0.7467,
    "f1": 0.6115
  },
  "tool_call_schema_accuracy": {
    "count_finish_reason_tool_calls": 985,
    "count_successful_tool_call": 985,
    "schema_accuracy": "100.00%"
  }
}
```

### Metric definitions

**ToolCall-Trigger Similarity**

| Metric | Formula | Meaning |
|--------|---------|---------|
| TP | — | Both vendor & baseline have `finish_reason == "tool_calls"` |
| FP | — | Vendor triggers tool_calls but baseline does not |
| FN | — | Baseline triggers tool_calls but vendor does not |
| TN | — | Neither triggers tool_calls |
| precision | `TP / (TP + FP)` | Of vendor-triggered tool calls, how many should have been triggered |
| recall | `TP / (TP + FN)` | Of tool calls that should have been triggered, how many were |
| **f1** | `2 * P * R / (P + R)` | **Primary metric for deployment correctness** |

**ToolCall-Schema Accuracy**

| Metric | Formula | Meaning |
|--------|---------|---------|
| count_finish_reason_tool_calls | — | Vendor responses with `finish_reason == "tool_calls"` |
| count_successful_tool_call | — | Tool call responses that passed JSON schema validation |
| **schema_accuracy** | `successful / total` | **Proportion of tool calls with valid schemas** |

## Adding a New Model Adapter

If a model requires special request preprocessing (e.g. role remapping, extra fields), create a new adapter:

1. Create `vendor_verifier/models/your_model.py`:

```python
from .base import ModelAdapter

class YourModel(ModelAdapter):
    model_patterns = ["your-model-name"]  # auto-match patterns

    def transform_messages(self, messages):
        # your custom transformations here
        return messages
```

2. Register it in `vendor_verifier/models/__init__.py`:

```python
from .your_model import YourModel

_REGISTRY.append(YourModel)
_TYPE_MAP["your-model"] = YourModel
```

For most OpenAI-compatible models, no adapter is needed — the `GenericModel` (default) works out of the box.

## Project Structure

```
vendor_verifier/
├── __init__.py        # Package exports
├── __main__.py        # Entry point for python -m vendor_verifier
├── cli.py             # CLI argument parsing
├── config.py          # Default constants
├── client.py          # Async API client with retry and streaming
├── validator.py       # Evaluation orchestration and statistics
├── validation.py      # JSON schema validation for tool calls
├── compare.py         # Compare vendor results against baseline (F1, schema accuracy)
└── models/
    ├── __init__.py    # Model registry and auto-detection
    ├── base.py        # ModelAdapter base class
    ├── generic.py     # Default adapter (OpenAI-compatible, no-op)
    └── kimi.py        # Kimi-specific adapter
```
