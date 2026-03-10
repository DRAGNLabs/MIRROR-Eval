# Multiturn Accuracy (MTA)

## About

Multiturn Accuracy (MTA) measures the accuracy of LLM responses over the course of a multiturn conversation.

MTA simulates a conversation with the model under test by iteratively passing it a prompt and follow-up responses, then saving the full conversation. It evaluates retention using LLM-as-a-judge with the prompt:

> "Rate, on a scale of 1 to 7, whether the information presented in the prompt is retained in the final response."

The judge score for each conversation is parsed from JSON, and analysis is run to compute summary statistics and percentile ranges.

## Requirements

- `accelerate` and Hugging Face Hub access
- The [`royal42/mta-test`](https://huggingface.co/datasets/royal42/mta-test) dataset (downloaded automatically)
- An available LLM for judgment

## Inputs

The following settings can be configured in `settings.toml`:

| Setting | Description |
|---|---|
| `metric` | Scoring approach — `"llm-as-a-judge"` (only option currently) |
| `dataset` | Conversation dataset to evaluate on |
| `judge_model` | HuggingFace model ID for the LLM judge |
| `prompt_type` | Judge prompt format — `"scale"` or `"category"` |
| `output_dir` | Directory to write results to |

## Outputs

MTA returns a JSON object with the following fields:

### Summary Statistics

| Field | Description |
|---|---|
| `total_examples` | Number of conversations evaluated |
| `mean` | Mean judge score |
| `std` | Standard deviation of scores |
| `median` | Median judge score (50th percentile) |
| `q1` | 25th percentile |
| `q3` | 75th percentile |
| `iqr` | Interquartile range — spread of the middle 50% of scores |

### Score Buckets

Scores on the 1–7 scale are grouped into three tiers:

| Field | Score Range | Description |
|---|---|---|
| `critical_failure_count`, `critical_failure_pct` | 1–2 | Model clearly failed to retain information |
| `mediocre_count`, `mediocre_pct` | 3–5 | Partial retention or low-quality response |
| `success_count`, `success_pct` | 6–7 | Model successfully retained information |
