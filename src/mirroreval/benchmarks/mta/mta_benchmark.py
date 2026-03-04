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

# Side-effect imports: ensure concrete classes register themselves
import mirroreval.benchmarks.mta.mta_metrics  # noqa: F401
import mirroreval.benchmarks.mta.mta_datasets  # noqa: F401
from mirroreval.benchmarks.mta.mta_analysis import compute_scores
from mirroreval.benchmarks.mta.mta_simulate_conversation import (
    simulate_conversation,
)
from mirroreval.logger import logger


def run_benchmark():
    # 1. Get the metrics and the datasets to run from settings
    metrics = settings.mta.metrics
    datasets = settings.mta.datasets

    # 2. Create output file for all results
    output_dir = Path(settings.mta.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_responses_file = output_dir / "mta_model_responses.jsonl"
    output_file = output_dir / "mta_results.jsonl"

    # Don't run if output file already exists
    if not output_file.exists():
        # 3. For each dataset, load the dataset
        for dataset_name in datasets:
            logger.info(f"Loading dataset: {dataset_name}")
            if dataset_name not in DATASETS:
                logger.error(
                    f"Error: Unknown dataset '{dataset_name}' specified in settings."
                )
                raise ValueError("Unknown dataset")

            dataset = DATASETS[dataset_name]()

            # 4. Run model through entire dataset, save out to intermediate results file
            simulate_conversation(dataset, model_responses_file)

        # 5. For each metric, run it through each dataset and save results to intermediate results file
        for metric_name in metrics:
            if metric_name not in METRICS:
                logger.error(
                    f"Error: Unknown metric '{metric_name}' specified in settings."
                )
                raise ValueError("Unknown metric")

            logger.info(f"Running metric: {metric_name}")

            METRICS[metric_name]()(model_responses_file)
            # Returns the metric computed for each line of all the datasets.
            # This is just line-wise scores, not corpus level stuff.

        # 6. Perform final analysis on all metric scores, corpus level stuff.
        final_scores = compute_scores(model_responses_file, output_file)

        return final_scores
    else:
        logger.info(
            f"Output file {output_file} already exists. Skipping benchmark run."
        )


if __name__ == "__main__":
    # Parse arguments for settings file path
    if len(sys.argv) > 1:
        settings_file_path = sys.argv[1]

        init_settings(settings_file_path)
    else:
        logger.error("No settings file path provided.")
        # Quit
        sys.exit(1)

    run_benchmark()
