"""
Creativity Analysis — Corpus-Level Aggregation
==============================================
Reads the pairwise creativity output JSONL file and aggregates it into a
single benchmark-level summary.

The pairwise file contains one record per ordered turn pair. This module
computes corpus-level statistics over those records, such as mean creativity,
mean persistence, and dataset coverage, then appends the summary as a JSON line
to the benchmark's summary output file.
"""

from __future__ import annotations

import json
import pandas as pd
from mirroreval.logger import logger


def load_jsonl(path):
    """Load a JSONL file into a pandas DataFrame. (one row per line)"""
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def compute_scores(input_path, output_file):
    """
    Aggregate pairwise benchmark rows into corpus-level summary statistics.

    Args:
        input_path: Path to the pairwise JSONL file written by the benchmark.
        output_file: Path to append the summary JSON object to.

    Returns:
        Dict containing corpus-level aggregate statistics.
    """
    df = load_jsonl(input_path)
    if not df.empty:
        logger.info(f"creativity_norm values: {df['creativity_norm']}")

    if df.empty:
        summary = {
            "total_pairs": 0,
            "total_conversations": 0,
        }
    else:
        summary = {
            "total_pairs": int(len(df)),
            "total_conversations": int(df["row_id"].nunique()),
            "mean_creativity_norm": round(float(df["creativity_norm"].mean()), 6),
            "median_creativity_norm": round(float(df["creativity_norm"].median()), 6),
            "std_creativity_norm": round(float(df["creativity_norm"].std(ddof=1)), 6)
            if len(df) > 1
            else 0.0,
            "min_creativity_norm": round(float(df["creativity_norm"].min()), 6),
            "max_creativity_norm": round(float(df["creativity_norm"].max()), 6),
            "mean_persistence": round(float(df["persistence"].mean()), 6),
            "mean_novelty": round(float(df["novelty"].mean()), 6),
            "mean_avg_max_sim": round(float(df["avg_max_sim"].mean()), 6),
            "pair_mode": df["pair_mode"].iloc[0],
            "mode": df["mode"].iloc[0],
            "role": df["role"].iloc[0],
            "datasets": sorted(df["dataset_name"].dropna().unique().tolist()),
        }

        if "persistence_same_position" in df.columns and not df[
            "persistence_same_position"
        ].isna().all():
            summary["mean_persistence_same_position"] = round(
                float(df["persistence_same_position"].mean()), 6
            )
        if "persistence_repositioned" in df.columns and not df[
            "persistence_repositioned"
        ].isna().all():
            summary["mean_persistence_repositioned"] = round(
                float(df["persistence_repositioned"].mean()), 6
            )
        if "avg_aligned_sim" in df.columns and not df["avg_aligned_sim"].isna().all():
            summary["mean_avg_aligned_sim"] = round(
                float(df["avg_aligned_sim"].mean()), 6
            )

        logger.info(
            "mean values: creativity_norm=%s, persistence=%s, novelty=%s, avg_max_sim=%s",
            summary["mean_creativity_norm"],
            summary["mean_persistence"],
            summary["mean_novelty"],
            summary["mean_avg_max_sim"],
        )
    # Append (not overwrite) — allows multiple benchmark runs to accumulate
    # results in the same file for comparison.
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")

    return summary
