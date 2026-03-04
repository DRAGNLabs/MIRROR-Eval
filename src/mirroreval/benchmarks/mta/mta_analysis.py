from mirroreval.config import settings
import json
import pandas as pd
from pathlib import Path
from scipy.stats import binomtest
from mirroreval.benchmarks.creativity.prompts import get_prompt_names
from mirroreval.logger import logger


def load_jsonl(path):
    """Load a JSONL file into a pandas DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def compute_scores(input_path, output_file):
    df = load_jsonl(input_path)

    # save accuracy, confidence intervals, and p-values to a json file
    accuracy_results = {
        # TODO: save relevant scores and metadata here
        "accuracy": 0
    }

    with open(output_file, "a") as f:
        for result in accuracy_results:
            f.write(json.dumps(result) + "\n")

    return accuracy_results
