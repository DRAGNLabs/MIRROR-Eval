"""
Creativity Metrics — Embedding-Based Turn Comparisons
=====================================================
Implements the pairwise metric logic used by the creativity benchmark.

These metrics were ported from MIRROR-CAP and operate over embeddings rather
than raw text generations. The benchmark measures how much content persists
from one turn to the next, how much appears novel, and derives a composite
creativity score from those signals.

Two operating modes are supported:
  1. sentence mode: each turn may contain multiple sentence embeddings, and
     the metric compares turns using max-over-sentence similarity
  2. message mode: each turn is represented by exactly one embedding vector

The registered metric class, EmbeddingCreativity, is a thin wrapper around the
core `compute_per_turn_metrics()` function so the benchmark can invoke it
through MIRROR-Eval's standard metric registry.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from mirroreval.benchmarks.interfaces import MetricInterface, register_metric


def _composite_creativity(
    persistence: float,
    avg_max_sim: float,
) -> Tuple[float, float]:
    """
    Combine persistence and similarity into a normalized creativity score.

    The metric treats creativity as the inverse pressure of two signals:
      - persistence: how much earlier content survives into later turns
      - similarity: how close the later turn remains to the previous turn

    Returns:
        Tuple `(creativity_raw, creativity_norm)` where the normalized value is
        scaled into the range `[0, 1]`.
    """
    anti_persistence_component = float(np.clip(1.0 - persistence, 0.0, 1.0))
    anti_similarity_component = float(np.clip((1.0 - avg_max_sim) / 2.0, 0.0, 1.0))
    creativity_raw = anti_persistence_component + anti_similarity_component
    creativity_norm = creativity_raw / 2.0
    return float(creativity_raw), float(creativity_norm)


def _rowwise_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between aligned rows of two matrices."""
    if a.shape != b.shape:
        raise ValueError("Aligned cosine similarity expects matching shapes.")

    numerators = np.sum(a * b, axis=1)
    denominators = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)) + 1e-12
    return numerators / denominators


def generate_turn_id_pairs(
    turn_ids: List[int],
    *,
    pair_mode: Literal["all", "sequential"] = "all",
) -> List[Tuple[int, int]]:
    """
    Generate ordered turn pairs for a single conversation.

    Args:
        turn_ids: Turn ids observed in the conversation.
        pair_mode:
            - `"all"` compares every ordered pair with increasing turn id
            - `"sequential"` compares adjacent turns only

    Returns:
        Ordered `(prev_turn_id, curr_turn_id)` pairs.
    """
    sorted_turn_ids = sorted({int(turn_id) for turn_id in turn_ids})
    if len(sorted_turn_ids) < 2:
        return []

    if pair_mode == "sequential":
        return [
            (sorted_turn_ids[index - 1], sorted_turn_ids[index])
            for index in range(1, len(sorted_turn_ids))
        ]

    return [
        (sorted_turn_ids[left], sorted_turn_ids[right])
        for left in range(len(sorted_turn_ids))
        for right in range(left + 1, len(sorted_turn_ids))
    ]


def build_turn_sentence_index(
    row_ids: List[int],
    turn_ids: List[int],
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, set[int]]]:
    """
    Group embedding row positions by conversation row id and turn id.

    This index lets the benchmark recover the set of embeddings belonging to a
    particular turn inside a particular conversation after the raw dataset has
    been exploded into many sentence or turn rows.
    """
    idx_by_row_turn: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    turn_ids_by_row: Dict[int, set[int]] = defaultdict(set)

    for idx, (row_id, turn_id) in enumerate(zip(row_ids, turn_ids)):
        rid = int(row_id)
        tid = int(turn_id)
        idx_by_row_turn[(rid, tid)].append(idx)
        turn_ids_by_row[rid].add(tid)

    return idx_by_row_turn, turn_ids_by_row


def turn_pair_metrics(
    prev_idx: List[int],
    curr_idx: List[int],
    embeddings: np.ndarray,
    *,
    threshold: float = 0.85,
) -> Dict[str, float]:
    """
    Compute sentence-level metrics between two ordered turns.

    Args:
        prev_idx: Embedding row indices for the earlier turn.
        curr_idx: Embedding row indices for the later turn.
        embeddings: Full embedding matrix aligned with `prev_idx`/`curr_idx`.
        threshold: Cosine-similarity cutoff used for persistence/novelty.

    Returns:
        Dictionary containing persistence, novelty, similarity, and creativity
        statistics for this ordered turn pair.
    """
    if not prev_idx or not curr_idx:
        return {
            "persistence": float(0.0 if prev_idx else np.nan),
            "persistence_same_position": float(0.0 if prev_idx else np.nan),
            "persistence_repositioned": float(0.0 if prev_idx else np.nan),
            "novelty": float(0.0 if curr_idx else np.nan),
            "avg_max_sim": float(np.nan),
            "avg_aligned_sim": float(np.nan),
            "creativity_raw": float(np.nan),
            "creativity_norm": float(np.nan),
            "matched_curr_ratio": float(np.nan),
            "n_prev": float(len(prev_idx)),
            "n_curr": float(len(curr_idx)),
        }

    prev_embeddings = embeddings[prev_idx]
    curr_embeddings = embeddings[curr_idx]

    sims = curr_embeddings @ prev_embeddings.T
    curr_to_prev = sims.max(axis=1)
    prev_to_curr = sims.max(axis=0)
    overlap = min(len(prev_idx), len(curr_idx))

    if overlap:
        aligned_sims = _rowwise_cosine_similarity(
            curr_embeddings[:overlap],
            prev_embeddings[:overlap],
        )
        same_position_matches = aligned_sims >= threshold
        avg_aligned_sim = float(aligned_sims.mean())
    else:
        same_position_matches = np.zeros(0, dtype=bool)
        avg_aligned_sim = float(np.nan)

    novelty = float((curr_to_prev < threshold).mean())
    persistence = float((prev_to_curr >= threshold).mean())
    avg_max_sim = float(curr_to_prev.mean())
    matched_curr = curr_to_prev >= threshold
    matched_curr_ratio = float(matched_curr.mean())
    same_position_persistence = float(same_position_matches.sum() / len(prev_idx))
    repositioned_persistence = float(
        max(persistence - same_position_persistence, 0.0)
    )
    effective_persistence = float((persistence + same_position_persistence) / 2.0)

    creativity_raw, creativity_norm = _composite_creativity(
        persistence=effective_persistence,
        avg_max_sim=avg_max_sim,
    )

    return {
        "persistence": persistence,
        "persistence_same_position": same_position_persistence,
        "persistence_repositioned": repositioned_persistence,
        "novelty": novelty,
        "avg_max_sim": avg_max_sim,
        "avg_aligned_sim": avg_aligned_sim,
        "creativity_raw": creativity_raw,
        "creativity_norm": creativity_norm,
        "matched_curr_ratio": matched_curr_ratio,
        "n_prev": float(len(prev_idx)),
        "n_curr": float(len(curr_idx)),
    }


def turn_pair_metrics_whole_message(
    prev_embedding: np.ndarray,
    curr_embedding: np.ndarray,
    *,
    threshold: float = 0.85,
) -> Dict[str, float]:
    """
    Compute message-level metrics between two whole-turn embeddings.

    This is the simpler variant of the metric used when the benchmark runs in
    `mode="message"` and there is exactly one embedding per turn.
    """
    prev = np.asarray(prev_embedding, dtype=np.float32)
    curr = np.asarray(curr_embedding, dtype=np.float32)

    denom = (np.linalg.norm(curr) * np.linalg.norm(prev)) + 1e-12
    sim = float(np.dot(curr, prev) / denom)
    persistence = float(sim >= threshold)
    novelty = float(sim < threshold)
    creativity_raw, creativity_norm = _composite_creativity(
        persistence=persistence,
        avg_max_sim=sim,
    )

    return {
        "persistence": persistence,
        "persistence_same_position": persistence,
        "persistence_repositioned": 0.0,
        "novelty": novelty,
        "avg_max_sim": float(sim),
        "avg_aligned_sim": float(sim),
        "creativity_raw": creativity_raw,
        "creativity_norm": creativity_norm,
        "matched_curr_ratio": persistence,
        "n_prev": 1.0,
        "n_curr": 1.0,
    }


def compute_per_turn_metrics(
    embeddings: np.ndarray,
    row_ids: List[int],
    turn_ids: List[int],
    *,
    threshold: float = 0.85,
    max_items: Optional[int] = None,
    pair_mode: Literal["all", "sequential"] = "all",
    mode: Literal["sentence", "message"] = "sentence",
) -> List[Dict[str, float]]:
    """
    Compute creativity metrics across all conversations in a dataset.

    Args:
        embeddings: Embedding matrix aligned with `row_ids` and `turn_ids`.
        row_ids: Conversation ids for each embedding row.
        turn_ids: Turn ids for each embedding row.
        threshold: Similarity cutoff used by the metric.
        max_items: Optional cap for smoke tests or debugging.
        pair_mode: Whether to compare all ordered turn pairs or only adjacent
            turns.
        mode: `"sentence"` or `"message"`.

    Returns:
        A list of metric rows, one per selected ordered turn pair.
    """
    if len(embeddings) != len(row_ids) or len(embeddings) != len(turn_ids):
        size = min(len(embeddings), len(row_ids), len(turn_ids))
        embeddings = embeddings[:size]
        row_ids = row_ids[:size]
        turn_ids = turn_ids[:size]

    if max_items is not None:
        embeddings = embeddings[:max_items]
        row_ids = row_ids[:max_items]
        turn_ids = turn_ids[:max_items]

    idx_by_row_turn, turn_ids_by_row = build_turn_sentence_index(row_ids, turn_ids)

    if mode == "message":
        for indices in idx_by_row_turn.values():
            if len(indices) != 1:
                raise ValueError(
                    "mode='message' expects exactly one embedding per (row_id, turn_id)."
                )

    metrics: List[Dict[str, float]] = []
    for row_id, tids in turn_ids_by_row.items():
        for prev_turn_id, curr_turn_id in generate_turn_id_pairs(
            list(tids),
            pair_mode=pair_mode,
        ):
            prev_idx = idx_by_row_turn[(row_id, prev_turn_id)]
            curr_idx = idx_by_row_turn[(row_id, curr_turn_id)]
            if mode == "message":
                values = turn_pair_metrics_whole_message(
                    embeddings[prev_idx[0]],
                    embeddings[curr_idx[0]],
                    threshold=threshold,
                )
            else:
                values = turn_pair_metrics(
                    prev_idx,
                    curr_idx,
                    embeddings,
                    threshold=threshold,
                )
            metrics.append(
                {
                    "row_id": int(row_id),
                    "prev_turn_id": int(prev_turn_id),
                    "curr_turn_id": int(curr_turn_id),
                    **values,
                }
            )

    return metrics


@register_metric("embedding-creativity")
class EmbeddingCreativity(MetricInterface):
    """
    Registry wrapper for the embedding-based creativity metric.

    MIRROR-Eval's metric registry expects callable classes. This class simply
    forwards the benchmark-provided embedding data into
    `compute_per_turn_metrics()`.
    """

    def __call__(
        self,
        dataset=None,
        *,
        embeddings: np.ndarray,
        row_ids: List[int],
        turn_ids: List[int],
        threshold: float = 0.85,
        max_items: Optional[int] = None,
        pair_mode: Literal["all", "sequential"] = "all",
        mode: Literal["sentence", "message"] = "sentence",
    ):
        """
        Compute turn-pair metrics for a pre-embedded benchmark dataset.

        The `dataset` positional argument is unused here because the creativity
        benchmark computes embeddings before invoking the metric. It is kept in
        the signature for compatibility with the shared MetricInterface.
        """
        del dataset
        return compute_per_turn_metrics(
            embeddings=embeddings,
            row_ids=row_ids,
            turn_ids=turn_ids,
            threshold=threshold,
            max_items=max_items,
            pair_mode=pair_mode,
            mode=mode,
        )
