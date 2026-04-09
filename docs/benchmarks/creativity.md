# Creativity

## About

Creativity is an embedding-based multiturn benchmark.
It measures how much information persists across turns, how much is novel, and
derives a composite creativity score from those signals.

The benchmark reads a prompt/archetype dataset, normalizes each example into an
ordered prompt plus follow-up turns, simulates assistant responses with the
configured model, materializes `P*`/`R*` turns, then extracts either user or
assistant turns, optionally splits turns into sentences, embeds the resulting
text, and computes ordered turn-pair metrics.

The recommended prompt-only dataset schema matches the MTA test shape:
`prompt`, `followup_1`, `followup_2`, `followup_3`. The default lightweight
dataset for this benchmark is `jackwarner/creativity-test`.

## Requirements

- Hugging Face Hub access for the dataset and embedding model
- A dataset that can be normalized into ordered user turns
- Supported input schemas include MTA-style `prompt` plus `followup_n` fields
  and MIRROR-CAP-style numbered `P*` columns
- `spaCy` is optional; if unavailable, sentence mode falls back to regex-based
  splitting

## Inputs

The following settings can be configured in `settings.toml`:

| Setting | Description |
|---|---|
| `metrics` | Metric registry names to run. Currently `"embedding-creativity"` |
| `datasets` | Hugging Face datasets that provide prompt/archetype examples such as `jackwarner/creativity-test` |
| `role` | Which conversation role to score: `"assistant"` or `"user"` |
| `mode` | `"sentence"` or `"message"` |
| `pair_mode` | `"all"` or `"sequential"` ordered turn pairs |
| `threshold` | Cosine-similarity cutoff used for persistence and novelty |
| `embedding_model` | Embedding model id or shorthand preset like `"minilm"` |
| `batch_size` | Embedding batch size |
| `max_length` | Token truncation length for embedding |
| `normalize_embeddings` | Whether to L2-normalize embeddings |
| `max_items` | Optional cap on embedded rows for smoke tests |
| `output_dir` | Directory for pairwise results and summary outputs |

## Outputs

The benchmark writes two files:

- `creativity_model_responses.jsonl`: one JSON object per simulated conversation in MTA-style `prompt`/`followup_n`/`response_n` format
- `creativity_pairwise_results.jsonl`: one JSON object per ordered turn pair
- `creativity_summary.jsonl`: one JSON object with corpus-level aggregates

Each pairwise result includes:

- `row_id`, `prev_turn_id`, `curr_turn_id`
- `persistence`, `persistence_same_position`, `persistence_repositioned`
- `novelty`, `avg_max_sim`, `avg_aligned_sim`
- `creativity_raw`, `creativity_norm`
- dataset and config metadata

The summary output includes:

- counts of total pairs and conversations
- mean, median, std, min, and max for `creativity_norm`
- mean persistence, novelty, and similarity statistics
- the benchmark mode, pair mode, role, and dataset names used
