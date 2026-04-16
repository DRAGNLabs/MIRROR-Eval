"""
MTA Analysis — Summary Score Computation
=========================================
Takes the raw final_results dict from run_benchmark() and computes
summary scores: probe quality, persistence (S3 + T-aggregates),
and control fire rates.

Output is a JSON-serializable dict saved alongside mta_results.json
as mta_summary.json.
"""

import numpy as np
from collections import defaultdict
from datetime import datetime, timezone


# ── Defaults ────────────────────────────────────────────────────────
EXCLUDE_LAYER_0 = True
TOP_K = 5
TAU = 0.5


def _r(x):
    """Round a float to 3 decimal places."""
    return round(float(x), 3)


def compute_turn_score(probe_predictions, probe_accuracies, exclude_layer_0=True):
    """Compute S3 (chance-baselined weighted) score for a single turn.

    Args:
        probe_predictions: dict {layer_str: bool} from one turn record.
        probe_accuracies: dict {layer_str: float} from probe_test_scores.
        exclude_layer_0: skip the embedding layer (key "0").

    Returns:
        float in [0, 1].
    """
    w_sum = 0.0
    wp_sum = 0.0
    for layer_str, pred in probe_predictions.items():
        if exclude_layer_0 and layer_str == "0":
            continue
        acc = probe_accuracies.get(layer_str, 0.5)
        w = max(0.0, acc - 0.5)
        w_sum += w
        wp_sum += w * (1.0 if pred else 0.0)
    return wp_sum / w_sum if w_sum > 0 else 0.0


def compute_conversation_scores(turn_scores, tau=TAU):
    """Compute T-aggregates from an ordered list of per-turn scores.

    Args:
        turn_scores: list of floats, one per turn, ordered by turn index.
        tau: threshold for T_first_drop.

    Returns:
        dict with T_mean, T_auc, T_decay, T_first_drop.
    """
    scores = np.asarray(turn_scores, dtype=float)
    n = len(scores)

    t_mean = float(scores.mean())
    t_auc = float(scores.sum())

    weights = np.arange(1, n + 1, dtype=float)
    t_decay = float(np.average(scores, weights=weights))

    # Largest 0-indexed turn t where all scores[0..t] >= tau.
    # Returns -1 if even turn 0 is below tau.
    t_first_drop = -1
    for i, s in enumerate(scores):
        if s >= tau:
            t_first_drop = i
        else:
            break

    return {
        "T_mean": _r(t_mean),
        "T_auc": _r(t_auc),
        "T_decay": _r(t_decay),
        "T_first_drop": t_first_drop,
    }


def _compute_fire_rates(turns, exclude_layer_0=True):
    """Compute fire rate stats for a list of turn records.

    Returns dict with mean_fire_rate, fire_rate_by_turn, fire_rate_by_layer.
    """
    by_turn = defaultdict(lambda: {"fires": 0, "total": 0})
    by_layer = defaultdict(lambda: {"fires": 0, "total": 0})
    total_fires = 0
    total_preds = 0

    for t in turns:
        turn_idx = t["turn"]
        for layer_str, pred in t["probe_predictions"].items():
            if exclude_layer_0 and layer_str == "0":
                continue
            val = 1 if pred else 0
            by_turn[turn_idx]["fires"] += val
            by_turn[turn_idx]["total"] += 1
            by_layer[layer_str]["fires"] += val
            by_layer[layer_str]["total"] += 1
            total_fires += val
            total_preds += 1

    mean_fire = total_fires / total_preds if total_preds > 0 else 0.0

    fire_by_turn = {
        str(k): _r(v["fires"] / v["total"]) if v["total"] > 0 else 0.0
        for k, v in sorted(by_turn.items())
    }
    fire_by_layer = {
        str(k): _r(v["fires"] / v["total"]) if v["total"] > 0 else 0.0
        for k, v in sorted(by_layer.items(), key=lambda x: int(x[0]))
    }

    return {
        "mean_fire_rate": _r(mean_fire),
        "fire_rate_by_turn": fire_by_turn,
        "fire_rate_by_layer": fire_by_layer,
    }


def analyze_results(final_results, exclude_layer_0=EXCLUDE_LAYER_0, tau=TAU):
    """Compute summary scores from raw MTA benchmark results.

    Args:
        final_results: the dict produced by run_benchmark().
        exclude_layer_0: skip embedding layer in aggregates.
        tau: threshold for T_first_drop.

    Returns:
        JSON-serializable summary dict.
    """
    meta = final_results["metadata"]
    pts = meta["probe_test_scores"]
    turns = final_results["turns"]

    # ── A. Metadata ─────────────────────────────────────────────────
    summary_meta = {
        "model": meta["model"],
        "activation_source": meta["probe_activation_source"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "exclude_layer_0": exclude_layer_0,
            "score_default": "S3",
            "tau": tau,
        },
    }

    # ── B. Probe quality ────────────────────────────────────────────
    probe_quality = {}
    for probe_type in ("real", "shuffled"):
        type_scores = pts.get(probe_type, {})
        per_fact = {}
        fact_means = []
        for fact_id, layers in type_scores.items():
            per_layer = {}
            accs = []
            for layer_str, acc in layers.items():
                per_layer[layer_str] = _r(acc)
                if exclude_layer_0 and layer_str == "0":
                    continue
                accs.append(acc)
            fact_mean = float(np.mean(accs)) if accs else 0.0
            per_fact[fact_id] = {
                "mean_accuracy": _r(fact_mean),
                "per_layer": per_layer,
            }
            fact_means.append(fact_mean)

        probe_quality[probe_type] = {
            "mean_accuracy": _r(np.mean(fact_means)) if fact_means else 0.0,
            "per_fact": per_fact,
        }

    # ── C. Persistence scores ───────────────────────────────────────
    # Filter to real, non-control turns
    real_turns = [
        t for t in turns
        if t.get("probe_type") == "real" and not t.get("is_control", False)
    ]

    # Group by (fact_id, pair_id) → list of turns
    convos = defaultdict(list)
    for t in real_turns:
        convos[(t["fact_id"], t["pair_id"])].append(t)

    # Per-conversation T-aggregates
    fact_convos = defaultdict(list)  # fact_id → list of T-aggregate dicts
    for (fact_id, pair_id), conv_turns in convos.items():
        conv_turns.sort(key=lambda t: t["turn"])
        accs = pts["real"].get(fact_id, {})
        turn_scores = [
            compute_turn_score(t["probe_predictions"], accs, exclude_layer_0)
            for t in conv_turns
        ]
        t_aggs = compute_conversation_scores(turn_scores, tau=tau)
        fact_convos[fact_id].append(t_aggs)

    # Per-fact averages
    per_fact_persistence = {}
    t_keys = ["T_mean", "T_auc", "T_decay", "T_first_drop"]
    all_fact_means = {k: [] for k in t_keys}

    for fact_id in sorted(fact_convos.keys()):
        aggs_list = fact_convos[fact_id]
        fact_avg = {}
        for k in t_keys:
            vals = [a[k] for a in aggs_list]
            m = float(np.mean(vals))
            fact_avg[k] = _r(m)
            all_fact_means[k].append(m)
        per_fact_persistence[fact_id] = fact_avg

    # Global summary (mean across facts)
    summary_persistence = {
        k: _r(np.mean(v)) if v else 0.0 for k, v in all_fact_means.items()
    }

    persistence_scores = {
        "summary": summary_persistence,
        "per_fact": per_fact_persistence,
    }

    # ── D. Controls ─────────────────────────────────────────────────
    real_baseline = [
        t for t in turns
        if t.get("probe_type") == "real" and not t.get("is_control", False)
    ]
    n1_turns = [
        t for t in turns
        if t.get("probe_type") == "real" and t.get("is_control", False)
    ]
    n2_turns = [
        t for t in turns
        if t.get("probe_type") == "shuffled" and not t.get("is_control", False)
    ]

    controls = {
        "real_baseline": _compute_fire_rates(real_baseline, exclude_layer_0),
        "N1_cross_fact": _compute_fire_rates(n1_turns, exclude_layer_0),
        "N2_shuffled": _compute_fire_rates(n2_turns, exclude_layer_0),
    }

    return {
        "metadata": summary_meta,
        "probe_quality": probe_quality,
        "persistence_scores": persistence_scores,
        "controls": controls,
    }
