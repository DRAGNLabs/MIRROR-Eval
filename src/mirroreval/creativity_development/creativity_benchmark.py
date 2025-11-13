import json
from pathlib import Path
import pandas as pd
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import binomtest
import sys
from enum import Enum

from mirroreval.config import init_settings, settings
from mirroreval.creativity_development.prompts import (
    get_prompt_names,
)
from mirroreval.creativity_development.creativity_metrics import (
    METRICS,
)
from mirroreval.creativity_development.creativity_datasets import (
    DATASETS,
)
from mirroreval.creativity_development.creativity_analysis import compute_scores
from mirroreval.logger import logger


def run_benchmark():
    # TODO:
    # figure out input/output definitions; you don't want each metric to have it's seperate line
    # figure out when the scores are computed (you should compute individual accuracy just when it happens, and then bigger stats afterword)
    # add yielding logic here for processing batches from what the metrics yield.
    # Defining a good interface for input/output of metrics/datasets will be important for having clean output.
    # 1. Get the metrics and the datasets to run from settings
    metrics = settings.creativity.metrics
    datasets = settings.creativity.datasets

    # 2. Create output file for all results
    output_dir = Path(settings.creativity.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.jsonl"

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

            for metric_name in metrics:
                if metric_name not in METRICS:
                    logger.error(
                        f"Error: Unknown metric '{metric_name}' specified in settings."
                    )
                    raise ValueError("Unknown metric")

                logger.info(f"Running metric: {metric_name} on dataset: {dataset_name}")

                results = METRICS[metric_name]()(dataset)

                with open(output_file, "a") as f:
                    for result in results:
                        result["dataset"] = dataset_name
                        result["metric"] = metric_name
                        f.write(json.dumps(result) + "\n")
    else:
        logger.info(
            f"Output file {output_file} already exists. Skipping benchmark run."
        )
    # After generating all results, compute scores and accuracy
    # TODO: implement automatic scoring
    # Need to define input/output schema/interface for all pieces of this
    # compute_scores(output_file)


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
