"""
MTA Benchmark — Main Orchestrator
==================================
This module coordinates the full MTA benchmark pipeline:

  1. Load datasets (via the registry)
  2. Simulate multi-turn conversations with the model under test
  3. Score each conversation using registered metrics (e.g., LLM-as-a-judge)
  4. Compute corpus-level analysis (mean, percentiles, score buckets)
  5. Save results

This file also serves as the __main__ entry point when launched via SLURM,
which runs it as a standalone script with the settings file path as an argument.
"""

import json
from pathlib import Path
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest
import sys
from enum import Enum

from mirroreval.config import init_settings, settings
from mirroreval.benchmarks.interfaces import METRICS, DATASETS
from mirroreval.benchmarks.mta.prompts import (
    get_prompt_names,
)

# --- Side-effect imports ---
# These imports look unused, but they are critical. When Python imports these
# modules, the @register_dataset and @register_metric decorators execute,
# which adds the concrete classes to the global DATASETS and METRICS dicts.
# Without these imports, DATASETS["royal42/mta-test"] and
# METRICS["llm-as-a-judge"] would not exist and the benchmark would fail.
import mirroreval.benchmarks.mta.mta_metrics  # noqa: F401
import mirroreval.benchmarks.mta.mta_datasets  # noqa: F401

from mirroreval.benchmarks.mta.mta_analysis import compute_scores
from mirroreval.benchmarks.mta.mta_simulate_conversation import (
    simulate_conversation,
)
from mirroreval.logger import logger


def run_benchmark():
    # --- Step 1: Read config ---
    # Pull the list of metrics and datasets from the [mta] section of settings.toml.
    # These are string identifiers that map to registered classes in the METRICS
    # and DATASETS dicts (populated by the side-effect imports above).
    metrics = settings.mta.metrics
    datasets = settings.mta.datasets

    # --- Step 2: Set up output paths ---
    # Two output files:
    #   - model_responses_file: intermediate JSONL with per-example model outputs
    #     and metric scores. Each line is one conversation.
    #   - output_file: final JSONL with corpus-level summary statistics.
    output_dir = Path(settings.mta.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_responses_file = output_dir / "mta_model_responses.jsonl"
    output_file = output_dir / "mta_results.jsonl"

    # Skip if results already exist (avoids re-running expensive evaluations).
    if not output_file.exists():

        # --- Step 3: Generate model responses ---
        # For each dataset, instantiate the registered class and run the model
        # through every example. simulate_conversation() handles the multi-turn
        # interaction and writes the results to the intermediate JSONL file.
        for dataset_name in datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            if dataset_name not in DATASETS:
                logger.error(
                    f"Error: Unknown dataset '{dataset_name}' specified in settings."
                )
                raise ValueError("Unknown dataset")

            # DATASETS[name] is a class, not an instance — calling it with ()
            # creates an instance, which triggers load_data() in __init__.
            dataset = DATASETS[dataset_name]()

            # Run the model under test through every example in this dataset.
            # Results are appended to model_responses_file as JSONL.
            simulate_conversation(dataset, model_responses_file)

        # --- Step 4: Run metrics ---
        # Metrics operate on the intermediate JSONL file produced above.
        # Each metric reads the file, scores every line, and writes the scores
        # back into the same file (atomically, via a temp file).
        # After this step, each line in model_responses_file has been enriched
        # with metric-specific fields (e.g., "llm_as_a_judge_score").
        for metric_name in metrics:
            if metric_name not in METRICS:
                logger.error(
                    f"Error: Unknown metric '{metric_name}' specified in settings."
                )
                raise ValueError("Unknown metric")

            logger.info(f"Running metric: {metric_name}")

            # METRICS[name] is a class — instantiate it, then call it with the
            # file path. The metric processes every line and updates the file.
            METRICS[metric_name]()(model_responses_file)

        # --- Step 5: Corpus-level analysis ---
        # compute_scores reads the per-example scores from the intermediate file
        # and aggregates them into summary statistics (mean, median, percentiles,
        # score buckets). The summary is appended to output_file as a JSON line.
        final_scores = compute_scores(model_responses_file, output_file)

        logger.info(f"Final scores: {final_scores}")

        return final_scores
    else:
        logger.info(
            f"Output file {output_file} already exists. Skipping benchmark run."
        )


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
