"""
Creativity Benchmark — Main Orchestrator
========================================
This module coordinates the full embedding-based creativity benchmark:

  1. Load datasets through the benchmark registry
  2. Convert multiturn conversations into sentence- or turn-level rows
  3. Embed the selected text units using the configured embedding model
  4. Compute registered turn-pair metrics
  5. Save pairwise outputs and corpus-level summary statistics

This file also acts as the __main__ entry point when launched via SLURM.
In that mode, the settings file path is passed on the command line and must
be loaded before any benchmark code reads the global settings object.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mirroreval.benchmarks.creativity.creativity_analysis import compute_scores
from mirroreval.benchmarks.creativity.creativity_embedding_model import (
    TextEmbedder,
    resolve_embedding_model_name,
)
from mirroreval.benchmarks.creativity.creativity_message_processing import (
    build_sentence_rows_for_role,
    build_turn_rows_for_role,
)
from mirroreval.benchmarks.interfaces import DATASETS, METRICS
from mirroreval.config import init_settings, settings
from mirroreval.logger import logger

# --- Side-effect imports ---
# These imports look unused, but they are critical. When Python imports these
# modules, the @register_dataset and @register_metric decorators execute,
# which adds the concrete classes to the global DATASETS and METRICS dicts.
# Without these imports, DATASETS["jackwarner/multi-turn-conversations"] and
# METRICS["embedding-creativity"] would not exist and the benchmark would fail.
import mirroreval.benchmarks.creativity.creativity_datasets  # noqa: F401
import mirroreval.benchmarks.creativity.creativity_metrics  # noqa: F401


def _build_rows(dataset_dict):
    """
    Convert a raw multiturn dataset into rows suitable for embedding.

    The benchmark supports two structural modes:
      - sentence: split each turn into multiple sentence rows
      - message: keep one row per whole turn

    The selected role in settings.creativity.role determines whether prompt
    turns (`P*`) or response turns (`R*`) are extracted.

    Returns:
        Tuple of:
          - the processed Hugging Face split/dataset containing row ids
          - the ordered text values to embed
          - the name of the turn-id column that aligns with those texts
    """
    role = settings.creativity.role
    message_prefix = "R" if role == "assistant" else "P"
    text_key = f"{role}_text"
    turn_id_key = f"{role}_turn_id"

    if settings.creativity.mode == "sentence":
        rows = build_sentence_rows_for_role(
            dataset_dict,
            message_prefix=message_prefix,
            turns_key=f"{role}_turns",
            turn_ids_key=f"{role}_turn_ids",
            turn_text_out_key=text_key,
            turn_id_out_key=turn_id_key,
            sentence_out_key=f"{role}_sentence",
        )["train"]
        texts = rows[f"{role}_sentence"]
    else:
        rows = build_turn_rows_for_role(
            dataset_dict,
            message_prefix=message_prefix,
            turns_key=f"{role}_turns",
            turn_ids_key=f"{role}_turn_ids",
            turn_text_out_key=text_key,
            turn_id_out_key=turn_id_key,
        )["train"]
        texts = rows[text_key]

    return rows, texts, turn_id_key


def _write_jsonl(path: Path, records):
    """Append a sequence of dictionaries to a JSONL file."""
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def run_benchmark():
    """
    Run the creativity benchmark end to end.

    The benchmark reads its configuration from the `[creativity]` section of
    settings.toml. For each configured dataset it:
      1. Loads the dataset class from the global DATASETS registry
      2. Builds ordered rows for the configured role/mode
      3. Embeds those rows with the configured model
      4. Invokes each registered metric to compute turn-pair scores
      5. Writes enriched pairwise rows to `creativity_pairwise_results.jsonl`

    After all datasets are processed, the benchmark aggregates the pairwise
    outputs into a single summary JSON object written to
    `creativity_summary.jsonl`.

    Returns:
        Dict containing corpus-level summary statistics.
    """
    # --- Step 1: Read config ---
    # Pull the list of metrics and datasets from the [creativity] section of
    # settings.toml. These are string identifiers that map to registered
    # classes in the METRICS and DATASETS dicts (populated by the side-effect
    # imports above).
    metrics = settings.creativity.metrics
    datasets = settings.creativity.datasets

    # --- Step 2: Set up output paths ---
    # Two output files:
    #   - pairwise_results_file: intermediate JSONL with one line per ordered
    #     turn-pair comparison and its computed metric values
    #   - summary_file: final JSONL with corpus-level summary statistics
    output_dir = Path(settings.creativity.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_results_file = output_dir / "creativity_pairwise_results.jsonl"
    summary_file = output_dir / "creativity_summary.jsonl"

    if pairwise_results_file.exists():
        pairwise_results_file.unlink()
    if summary_file.exists():
        summary_file.unlink()

    # --- Step 3: Initialize the embedding model ---
    # The creativity benchmark uses an embedding model rather than a judge LLM.
    # We resolve shorthand config values such as "minilm" to their full
    # Hugging Face identifiers, then load the embedder once so it can be
    # reused across every configured dataset.
    resolved_model_name = resolve_embedding_model_name(
        settings.creativity.embedding_model
    )
    embedder = TextEmbedder(model_name=resolved_model_name)

    # --- Step 4: Load and process each dataset ---
    # For each configured dataset:
    #   1. Instantiate the registered dataset class
    #   2. Build either sentence-level or message-level rows, depending on the
    #      configured benchmark mode
    #   3. Embed the ordered text rows
    #   4. Run every registered metric over the resulting embeddings
    for dataset_name in datasets:
        logger.info(f"Loading dataset: {dataset_name}")
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset_name}' specified in settings.")

        dataset = DATASETS[dataset_name]()
        rows, texts, turn_id_key = _build_rows(dataset.dataset)

        # The rows returned by _build_rows() preserve row_id and turn_id
        # alignment so metric outputs can be traced back to their source
        # conversation and ordered turn pair.
        logger.info(
            "Embedding %s rows with %s",
            len(texts),
            resolved_model_name,
        )
        embeddings = embedder.embed(
            texts,
            batch_size=settings.creativity.batch_size,
            max_length=settings.creativity.max_length,
            normalize=settings.creativity.normalize_embeddings,
            show_progress=False,
        )

        # --- Step 5: Run metrics ---
        # Each metric receives the embedding matrix plus aligned row_id/turn_id
        # columns. The metric returns one result row per ordered turn pair
        # selected by the configured pair_mode.
        for metric_name in metrics:
            if metric_name not in METRICS:
                raise ValueError(f"Unknown metric '{metric_name}' specified in settings.")

            metric_rows = METRICS[metric_name]()(  # type: ignore[misc]
                embeddings=embeddings,
                row_ids=rows["row_id"],
                turn_ids=rows[turn_id_key],
                threshold=settings.creativity.threshold,
                max_items=settings.creativity.max_items,
                pair_mode=settings.creativity.pair_mode,
                mode=settings.creativity.mode,
            )

            enriched_rows = []
            for metric_row in metric_rows:
                enriched_rows.append(
                    {
                        **metric_row,
                        "dataset_name": dataset_name,
                        "metric_name": metric_name,
                        "role": settings.creativity.role,
                        "mode": settings.creativity.mode,
                        "pair_mode": settings.creativity.pair_mode,
                        "embedding_model": resolved_model_name,
                    }
                )

            # Append the enriched per-pair metric rows to the intermediate
            # JSONL file. Each output line corresponds to one ordered turn-pair
            # comparison, not one full conversation and not one sentence row.
            _write_jsonl(pairwise_results_file, enriched_rows)

    # --- Step 6: Corpus-level analysis ---
    # compute_scores reads the pairwise JSONL file and aggregates those
    # ordered turn-pair results into summary statistics for the full run.
    summary = compute_scores(pairwise_results_file, summary_file)
    logger.info(f"Final creativity scores: {summary}")
    return summary

# When launched via SLURM, this script is run directly with the settings file
# path as a command-line argument. init_settings() must be called before any
# code accesses the settings object.
if __name__ == "__main__":
    if len(sys.argv) > 1:
        settings_file_path = sys.argv[1]
        init_settings(settings_file_path)
    else:
        logger.error("No settings file path provided.")
        sys.exit(1)

    run_benchmark()
