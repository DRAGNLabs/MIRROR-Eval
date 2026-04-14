"""
MQM Benchmark — MT Metric Reliability
======================================
Evaluates a translation model by:

  1. Translating FLORES+ English sentences into configured target languages
     using the model under test (settings.model.model_checkpoint_path).
  2. Computing automated MT metrics (BLEU, ChrF, COMET) on those translations
     against FLORES+ reference translations.
  3. Reporting per-language metric scores alongside empirical metric reliability
     estimates derived from WMT human-judgment calibration data.

The reliability estimates tell the user how much to trust each metric's scores
for the evaluated model. They are derived from a multilingual meta-evaluation
study across 11 languages and three resource tiers (Spearman correlation and
pairwise accuracy vs. professional human MQM / Direct Assessment annotations).

Output JSON structure (saved to settings.mqm.output_dir/mqm_results.json):
  {
    "model": "<model_checkpoint_path>",
    "languages_evaluated": [...],
    "metric_scores": {
      "de": {"bleu": 45.2, "chrf": 60.1, "comet": 0.821},
      ...
    },
    "metric_reliability": {
      "bleu":  {"spearman_r": 0.107, "pairwise_acc": 0.544, "tier": "high-resource"},
      "chrf":  {"spearman_r": 0.143, "pairwise_acc": 0.560, "tier": "high-resource"},
      "comet": {"spearman_r": 0.343, "pairwise_acc": 0.639, "tier": "high-resource"},
      ...   (entries per resource tier)
    },
    "recommendation": "COMET is the most reliable automated signal for this language set..."
  }

The model_checkpoint_path is expected to be an NLLB-200-style seq2seq translation
model (e.g., "facebook/nllb-200-distilled-600M"). The model must support the
`translation` task via a HuggingFace transformers pipeline with src_lang /
tgt_lang arguments.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from mirroreval.config import init_settings, settings
from mirroreval.benchmarks.interfaces import METRICS, DATASETS
from mirroreval.logger import logger

# Side-effect imports — trigger @register_dataset / @register_metric decorators
import mirroreval.benchmarks.mqm.mqm_datasets  # noqa: F401
import mirroreval.benchmarks.mqm.mqm_metrics   # noqa: F401

from mirroreval.benchmarks.mqm.mqm_datasets import NLLB_LANG_CODES


# ---------------------------------------------------------------------------
# Pre-computed metric reliability calibration
# ---------------------------------------------------------------------------
# These values are from a multilingual meta-evaluation study over 371k+ segments
# across 11 languages using WMT MQM (professional) and WMT Direct Assessment
# human annotations. Three resource tiers are reported separately.
#
# Reference: "Multilingual MT Metric Meta-Evaluation" (Turley, BYU 2026)
# Methodology: Spearman ρ and pairwise accuracy (concordant pairs / total pairs,
#   max_n=8000 sample) computed per language, then averaged within each tier.
#   Williams (1959) tests confirm COMET significantly outperforms all other
#   metrics at p ≈ 0 for every language evaluated.

METRIC_RELIABILITY = {
    "bleu": {
        "high":   {"spearman_r": 0.107, "pairwise_acc": 0.544},
        "medium": {"spearman_r": 0.206, "pairwise_acc": 0.569},
        "low":    {"spearman_r": 0.230, "pairwise_acc": 0.576},
    },
    "chrf": {
        "high":   {"spearman_r": 0.143, "pairwise_acc": 0.560},
        "medium": {"spearman_r": 0.233, "pairwise_acc": 0.580},
        "low":    {"spearman_r": 0.267, "pairwise_acc": 0.590},
    },
    "comet": {
        "high":   {"spearman_r": 0.343, "pairwise_acc": 0.639},
        "medium": {"spearman_r": 0.378, "pairwise_acc": 0.633},
        "low":    {"spearman_r": 0.331, "pairwise_acc": 0.613},
    },
}

# Resource tier classification for languages supported by this benchmark.
# Languages not listed here are treated as "low".
RESOURCE_TIERS = {
    "de": "high", "zh": "high", "ru": "high", "he": "high",
    "es": "medium", "cs": "medium", "tr": "medium", "uk": "medium",
    "ha": "low", "km": "low", "ps": "low",
    "sw": "low", "ht": "low", "lo": "low",
}


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def _translate_with_model(sources: list[str], nllb_tgt_lang: str,
                           model_name: str, batch_size: int = 32) -> list[str]:
    """
    Translate a list of English source sentences to tgt_lang using model_name.

    The model is expected to be an NLLB-200-style seq2seq model that accepts
    src_lang / tgt_lang tokens (e.g., "facebook/nllb-200-distilled-600M").

    Returns a list of translated strings in the same order as sources.
    """
    from transformers import pipeline as hf_pipeline
    import torch

    logger.info(
        f"Translating {len(sources)} sentences to {nllb_tgt_lang} "
        f"with {model_name}..."
    )
    device = 0 if torch.cuda.is_available() else -1
    pipe = hf_pipeline(
        "translation",
        model=model_name,
        src_lang="eng_Latn",
        tgt_lang=nllb_tgt_lang,
        device=device,
        max_length=256,
    )
    results = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i: i + batch_size]
        outputs = pipe(batch)
        results.extend(o["translation_text"] for o in outputs)
    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def _corpus_metric_scores(segments: list[dict]) -> dict:
    """
    Average per-segment metric scores across all segments for a single language.
    Returns {"bleu": float, "chrf": float, "comet": float | None}.
    """
    scores = defaultdict(list)
    for seg in segments:
        for key in ("bleu_score", "chrf_score", "comet_score"):
            if key in seg and seg[key] is not None:
                scores[key].append(seg[key])
    return {
        "bleu":  sum(scores["bleu_score"])  / len(scores["bleu_score"])  if scores["bleu_score"]  else None,
        "chrf":  sum(scores["chrf_score"])  / len(scores["chrf_score"])  if scores["chrf_score"]  else None,
        "comet": sum(scores["comet_score"]) / len(scores["comet_score"]) if scores["comet_score"] else None,
    }


def _build_reliability_section(langs: list[str]) -> dict:
    """
    Build the metric_reliability block for the results JSON, reporting
    calibration numbers for each resource tier present in langs.
    """
    tiers_present = sorted({RESOURCE_TIERS.get(l, "low") for l in langs})
    reliability = {}
    for metric, tier_data in METRIC_RELIABILITY.items():
        reliability[metric] = {
            tier: tier_data[tier]
            for tier in tiers_present
            if tier in tier_data
        }
    return reliability


def _recommendation(langs: list[str]) -> str:
    tiers = {RESOURCE_TIERS.get(l, "low") for l in langs}
    tier_str = " / ".join(sorted(tiers))
    return (
        f"COMET (Unbabel/wmt22-comet-da) is the most reliable automated metric "
        f"for this language set ({tier_str} resource tier(s)), outperforming BLEU "
        f"and ChrF on Spearman correlation and pairwise accuracy vs. human judgment "
        f"at p ≈ 0 (Williams test). BLEU and ChrF provide fast reference-only "
        f"alternatives when COMET is unavailable, with ChrF preferred over BLEU "
        f"for morphologically complex or non-Latin-script languages."
    )


# ---------------------------------------------------------------------------
# Main benchmark entry point
# ---------------------------------------------------------------------------
def run_benchmark():
    model_name = settings.model.model_checkpoint_path
    target_langs = list(settings.mqm.languages)
    run_metrics = list(settings.mqm.metrics)
    batch_size = int(settings.mqm.get("translation_batch_size", 32))
    flores_split = str(settings.mqm.get("flores_split", "devtest"))

    output_dir = Path(settings.mqm.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mqm_results.json"

    if output_file.exists():
        logger.info(f"Results already exist at {output_file}. Skipping.")
        with open(output_file) as f:
            return json.load(f)

    # --- Load FLORES+ dataset ---
    logger.info("Loading FLORES+ dataset...")
    dataset = DATASETS["openlanguagedata/flores_plus"](
        target_langs=target_langs, split=flores_split
    )

    # Group segments by language
    segments_by_lang: dict[str, list[dict]] = defaultdict(list)
    for seg in dataset:
        segments_by_lang[seg["lang"]].append(seg)

    # --- Translate with model under test ---
    for lang, segs in segments_by_lang.items():
        nllb_lang = NLLB_LANG_CODES.get(lang)
        if nllb_lang is None:
            logger.warning(f"No NLLB code for '{lang}', skipping translation.")
            for seg in segs:
                seg["hypothesis"] = ""
            continue
        sources = [s["source"] for s in segs]
        hypotheses = _translate_with_model(
            sources, nllb_lang, model_name, batch_size=batch_size
        )
        for seg, hyp in zip(segs, hypotheses):
            seg["hypothesis"] = hyp

    # Flatten segments back to a list
    all_segments = [seg for segs in segments_by_lang.values() for seg in segs]

    # --- Run metrics ---
    for metric_name in run_metrics:
        if metric_name not in METRICS:
            logger.warning(f"Unknown metric '{metric_name}', skipping.")
            continue
        logger.info(f"Running metric: {metric_name}")
        all_segments = METRICS[metric_name]()(all_segments)

    # Re-group by language for per-language scoring
    segments_by_lang_scored: dict[str, list[dict]] = defaultdict(list)
    for seg in all_segments:
        segments_by_lang_scored[seg["lang"]].append(seg)

    # --- Build results ---
    metric_scores = {
        lang: _corpus_metric_scores(segs)
        for lang, segs in segments_by_lang_scored.items()
    }

    results = {
        "model": model_name,
        "languages_evaluated": target_langs,
        "flores_split": flores_split,
        "metrics_run": run_metrics,
        "metric_scores": metric_scores,
        "metric_reliability": _build_reliability_section(target_langs),
        "recommendation": _recommendation(target_langs),
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    logger.info(f"Metric scores: {metric_scores}")
    return results


# ---------------------------------------------------------------------------
# SLURM entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        init_settings(sys.argv[1])
    else:
        logger.error("No settings file path provided.")
        sys.exit(1)
    run_benchmark()
