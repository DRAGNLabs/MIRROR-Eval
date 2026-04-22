"""
MQM Metrics
===========
Implements MT metric scoring for the MQM benchmark.

Three metric classes are registered:
  - "bleu"       : BLEU (sacrebleu corpus-level)
  - "chrf"       : ChrF (sacrebleu corpus-level)
  - "comet"      : COMET-DA (Unbabel/wmt22-comet-da, reference-based)

Each class implements MetricInterface. When called with a list of segment dicts
(each containing "hypothesis", "reference", and optionally "source"), it adds
a score field to every dict and returns the updated list.

Usage pattern inside mqm_benchmark.py:
    segments = METRICS["comet"]()(segments)
    # each segment now has a "comet_score" field

Requires:
    pip install sacrebleu unbabel-comet
"""

from typing import Any

from mirroreval.benchmarks.interfaces import MetricInterface, register_metric
from mirroreval.logger import logger


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------
@register_metric("bleu")
class BLEUMetric(MetricInterface):
    """
    Sentence-level BLEU via sacrebleu.

    Adds "bleu_score" (float, 0–100) to each segment dict.
    """

    def __call__(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required: pip install sacrebleu")

        logger.info("Computing BLEU scores...")
        for seg in segments:
            hyp = seg.get("hypothesis", "")
            ref = seg.get("reference", "")
            try:
                score = sacrebleu.sentence_bleu(hyp, [ref]).score
            except Exception as exc:
                logger.warning(f"BLEU scoring failed for segment (hyp={hyp!r:.40}): {exc}")
                score = 0.0
            seg["bleu_score"] = score
        return segments


# ---------------------------------------------------------------------------
# ChrF
# ---------------------------------------------------------------------------
@register_metric("chrf")
class ChrFMetric(MetricInterface):
    """
    Sentence-level ChrF via sacrebleu.

    Adds "chrf_score" (float, 0–100) to each segment dict.
    """

    def __call__(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required: pip install sacrebleu")

        logger.info("Computing ChrF scores...")
        chrf = sacrebleu.CHRF()
        for seg in segments:
            hyp = seg.get("hypothesis", "")
            ref = seg.get("reference", "")
            try:
                score = chrf.sentence_score(hyp, [ref]).score
            except Exception as exc:
                logger.warning(f"ChrF scoring failed for segment (hyp={hyp!r:.40}): {exc}")
                score = 0.0
            seg["chrf_score"] = score
        return segments


# ---------------------------------------------------------------------------
# COMET
# ---------------------------------------------------------------------------
@register_metric("comet")
class COMETMetric(MetricInterface):
    """
    COMET-DA reference-based metric (Unbabel/wmt22-comet-da).

    Adds "comet_score" (float, roughly 0–1) to each segment dict.
    Requires the unbabel-comet package and model weights cached locally.

    Note: COMET model must be downloaded before running on a compute node
    that has HF_HUB_OFFLINE=1 set. The entrypoint handles this via
    download_from_hf().
    """

    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da", gpus: int = 0,
                 batch_size: int = 16):
        self.model_name = model_name
        self.gpus = gpus
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from comet import load_from_checkpoint, download_model
            except ImportError:
                raise ImportError("unbabel-comet is required: pip install unbabel-comet")
            logger.info(f"Loading COMET model: {self.model_name}")
            checkpoint = download_model(self.model_name)
            self._model = load_from_checkpoint(checkpoint)
        return self._model

    def __call__(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        model = self._load_model()
        data = [
            {
                "src": seg.get("source", ""),
                "mt": seg.get("hypothesis", ""),
                "ref": seg.get("reference", ""),
            }
            for seg in segments
        ]
        logger.info(f"Scoring {len(data)} segments with COMET...")
        output = model.predict(data, batch_size=self.batch_size, gpus=self.gpus)
        for seg, score in zip(segments, output.scores):
            seg["comet_score"] = float(score)
        return segments
