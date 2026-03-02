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


def compute_scores(input_path):
    df = load_jsonl(input_path)

    # Compute correct_label
    df["correct_label"] = (df["Diversity_Set1"] < df["Diversity_Set2"]).astype(int)
    ties = df["Diversity_Set1"] == df["Diversity_Set2"]
    df.loc[ties, "correct_label"] = -1

    # Parse output_both_sets into two columns
    both_scores = df["output_both_sets"].str.strip("()").str.split(",", expand=True)
    df["score_both_0"] = pd.to_numeric(both_scores[0], errors="coerce")
    df["score_both_1"] = pd.to_numeric(both_scores[1], errors="coerce")

    # Compute predicted_label_both
    df["predicted_label_both"] = (df["score_both_0"] < df["score_both_1"]).astype(int)
    ties_both = df["score_both_0"] == df["score_both_1"]
    df.loc[ties_both, "predicted_label_both"] = -1

    # Compute accuracy_both
    df["accuracy_both"] = (df["predicted_label_both"] == df["correct_label"]).astype(
        int
    )

    # Parse output_set_1 and output_set_2
    df["score_set_1"] = pd.to_numeric(
        df["output_set_1"].str.strip("()"), errors="coerce"
    )
    df["score_set_2"] = pd.to_numeric(
        df["output_set_2"].str.strip("()"), errors="coerce"
    )

    # Compute predicted_label_individual
    df["predicted_label_individual"] = (df["score_set_1"] < df["score_set_2"]).astype(
        int
    )
    ties_ind = df["score_set_1"] == df["score_set_2"]
    df.loc[ties_ind, "predicted_label_individual"] = -1

    # Compute accuracy_individual_set
    df["accuracy_individual_set"] = (
        df["predicted_label_individual"] == df["correct_label"]
    ).astype(int)

    # Compute accuracy and other scores.
    # We get accuracy for each prompt type for each model, on each dataset split.
    for pt in get_prompt_names():
        for model in df["model_name"].unique():
            for split in df["split_name"].unique():
                subset = df[
                    (df["prompt"] == pt)
                    & (df["model_name"] == model)
                    & (df["split_name"] == split)
                ]

                total = len(subset)
                correct_both = subset["accuracy_both"].sum()
                correct_individual = subset["accuracy_individual_set"].sum()

                accuracy_both = correct_both / total if total > 0 else 0
                accuracy_individual = correct_individual / total if total > 0 else 0

                result_both = binomtest(correct_both, total)
                result_individual = binomtest(correct_individual, total)

                both_ci_low, both_ci_high = result_both.proportion_ci(
                    confidence_level=0.95, method="exact"
                )
                individual_ci_low, individual_ci_high = result_individual.proportion_ci(
                    confidence_level=0.95, method="exact"
                )

                logger.info(f"Prompt type: {pt}")
                logger.info(f"Model: {model}")
                logger.info(f"Dataset split: {split}")
                logger.info(f"Total examples: {total}")
                logger.info(f"Accuracy (both sets): {accuracy_both:.4f}")
                logger.info(f"Accuracy (individual sets): {accuracy_individual:.4f}")
                logger.info(f"95% CI (both sets): ({both_ci_low:.4f}, {both_ci_high:.4f})")
                logger.info(
                    f"95% CI (individual sets): ({individual_ci_low:.4f}, {individual_ci_high:.4f})"
                )

                accuracy_both = round(accuracy_both, 3)
                accuracy_individual = round(accuracy_individual, 3)
                both_ci_low = round(both_ci_low, 3)
                both_ci_high = round(both_ci_high, 3)
                individual_ci_low = round(individual_ci_low, 3)
                individual_ci_high = round(individual_ci_high, 3)

                # save accuracy, confidence intervals, and p-values to a json file
                accuracy_results = {
                    "prompt_type": pt,
                    "model": model,
                    "dataset_split": split,
                    "total_examples": total,
                    "accuracy_both": accuracy_both,
                    "accuracy_individual": accuracy_individual,
                    "ci_both": [both_ci_low, both_ci_high],
                    "ci_individual": [individual_ci_low, individual_ci_high],
                }

                output_dir = Path(settings.creativity.output_dir)
                model_safe = model.replace("/", "_")
                with open(
                    output_dir / f"accuracy_{pt}_{model_safe}_{split}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(accuracy_results, f, ensure_ascii=False, indent=4)

                # Also append it to a master accuracy file
                with open(
                    output_dir / "accuracy_all.jsonl",
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(accuracy_results, ensure_ascii=False) + "\n")

    df.to_json(
        f"{Path(settings.creativity.output_dir)}/results_scored.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )
