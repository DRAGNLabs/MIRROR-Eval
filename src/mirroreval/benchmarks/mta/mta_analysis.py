"""
MTA Analysis — Corpus-Level Score Aggregation
===============================================
Reads per-example scores from the intermediate JSONL file and computes
corpus-level summary statistics.

This is the final step in the MTA pipeline. After simulate_conversation
generates responses and LLM-as-a-judge scores each one, this module
aggregates those individual scores into a summary that characterizes the
model's overall performance.

Output is appended to the results JSONL file as a single JSON line.
"""

from mirroreval.config import settings
import json
import numpy as np
import pandas as pd
from pathlib import Path
from mirroreval.logger import logger


def load_jsonl(path):
    """Load a JSONL file into a pandas DataFrame (one row per line)."""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def compute_scores(input_path, output_file):
    """
    Aggregate per-example LLM-as-a-judge scores into corpus-level statistics.

    Args:
        input_path: Path to the intermediate JSONL file containing per-example
                    scores (each line has an "llm_as_a_judge_score" field).
        output_file: Path to append the summary JSON line to.

    Returns:
        Dict with summary statistics.
    """
    # df = load_jsonl(input_path)
    # logger.info(f"scores: {df['llm_as_a_judge_score']}")
    # scores = df["llm_as_a_judge_score"].astype(float)

    # total = len(scores)

    # # --- Percentile statistics ---
    # # q1 (25th percentile), median (50th), q3 (75th) describe the distribution.
    # # IQR (interquartile range) = q3 - q1, measures the spread of the middle 50%.
    # q1, median, q3 = np.percentile(scores, [25, 50, 75])
    # iqr = q3 - q1

    # # --- Score buckets ---
    # # Group scores on the 1-7 scale into three interpretive tiers:
    # #   - Critical failure (1-2): model clearly failed to retain information
    # #   - Mediocre (3-5): partial retention or low-quality response
    # #   - Success (6-7): model successfully retained information
    # # Both counts and percentages are reported.
    # critical_failure = scores.between(1, 2).sum()
    # mediocre = scores.between(3, 5).sum()
    # success = scores.between(6, 7).sum()

    # accuracy_results = {
    #     "total_examples": total,
    #     "median": round(float(median), 3),
    #     "q1": round(float(q1), 3),
    #     "q3": round(float(q3), 3),
    #     "iqr": round(float(iqr), 3),
    #     "mean": round(scores.mean(), 3),  # type: ignore[arg-type]
    #     "std": round(scores.std(), 3),  # type: ignore[arg-type]
    #     "critical_failure_count": int(critical_failure),
    #     "critical_failure_pct": round(critical_failure / total * 100, 2),
    #     "mediocre_count": int(mediocre),
    #     "mediocre_pct": round(mediocre / total * 100, 2),
    #     "success_count": int(success),
    #     "success_pct": round(success / total * 100, 2),
    # }

    # # Append (not overwrite) — allows multiple benchmark runs to accumulate
    # # results in the same file for comparison.
    # with open(output_file, "a") as f:
    #     f.write(json.dumps(accuracy_results) + "\n")

    # return accuracy_results
    return
