"""
MTA Benchmark — Main Orchestrator
==================================
Coordinates the full MTA benchmark pipeline:

  1. Load model-under-test and partner model (once)
  2. For each fact: train probes on a held-out split, simulate conversations
     on the eval split, and record per-turn probe predictions
  3. Save results as a flat JSON structure for analysis

This file also serves as the __main__ entry point when launched via SLURM.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mirroreval.config import init_settings, settings
from mirroreval.benchmarks.mta.mta_datasets import MTA
from mirroreval.benchmarks.mta.mta_simulate_conversation import simulate_conversation
from mirroreval.benchmarks.mta.mta_probes import train_probes, apply_probes
from mirroreval.hf_utilities import has_chat_template
from mirroreval.partner import get_partner
from mirroreval.logger import logger


def run_benchmark():
    # --- Config ---
    model_checkpoint = settings.model.model_checkpoint_path
    partner_checkpoint = settings.mta.partner_checkpoint_path
    num_turns = settings.mta.num_conversation_turns

    output_dir = Path(settings.mta.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mta_results.json"
    progress_file = output_dir / "mta_progress.jsonl"

    if output_file.exists():
        logger.info(
            f"Output file {output_file} already exists. Skipping benchmark run."
        )
        return

    # --- Load models once ---
    model_quantize = settings.model.model_quantize or None
    logger.info(
        f"Loading model under test: {model_checkpoint} (quantize={model_quantize})"
    )

    model_kwargs = {"output_hidden_states": True, "device_map": "auto"}
    if model_quantize == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif model_quantize == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, **model_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    partner_backend = settings.mta.partner_backend
    partner_quantize = settings.mta.partner_quantize or None

    # Verify both models support chat templates before doing any work
    if not has_chat_template(model_checkpoint):
        raise ValueError(f"Model {model_checkpoint} does not support chat templates.")
    if partner_backend == "local" and not has_chat_template(partner_checkpoint):
        raise ValueError(
            f"Partner model {partner_checkpoint} does not support chat templates."
        )

    partner = get_partner(
        partner_backend, partner_checkpoint, quantize=partner_quantize
    )

    # --- Load dataset ---
    logger.info("Loading dataset")
    dataset = MTA()

    # --- Run pipeline ---
    all_turns = []
    activation_source = getattr(settings.mta, "probe_activation_source", "full_context")
    if activation_source not in ("full_context", "isolated_turn"):
        raise ValueError(
            f"Invalid probe_activation_source: {activation_source!r}. "
            "Must be 'full_context' or 'isolated_turn'."
        )
    logger.info(f"Probe activation source: {activation_source}")

    # Every run produces three kinds of records in one output file:
    #   - real: real probes applied to real eval conversations
    #   - N1 : real probes applied to a cross-fact control conversation
    #   - N2 : shuffled-label probes applied to the same conversations (both
    #          real and N1), as a null-control baseline
    # Records are distinguished by (probe_type, is_control).
    probe_test_scores = {"real": {}, "shuffled": {}}

    # Materialize the fact list so we can index for N1 control rotation.
    fact_ids = list(dataset.iter_fact_id())

    def _emit_records(
        probe_variants,
        activations_per_turn,
        conversation,
        *,
        fact_id,
        pair_id,
        fact_text,
        is_control,
        source_fact_id,
    ):
        for turn_idx, turn_acts in enumerate(activations_per_turn):
            for probe_type, probes in probe_variants:
                preds = apply_probes(probes, turn_acts)
                turn_record = {
                    "fact_id": fact_id,
                    "pair_id": pair_id,
                    "fact": fact_text,
                    "initial_prompt": conversation["initial_prompt"],
                    "turn": turn_idx,
                    "model_response": conversation["model_responses"][turn_idx],
                    "partner_response": conversation["partner_responses"][turn_idx],
                    "probe_predictions": {
                        str(k): bool(v) for k, v in preds.items()
                    },
                    "probe_type": probe_type,
                    "is_control": is_control,
                    "probe_fact_id": fact_id,
                    "source_fact_id": source_fact_id,
                }
                all_turns.append(turn_record)
                # Incremental progress — survives crashes
                with open(progress_file, "a") as f:
                    f.write(json.dumps(turn_record) + "\n")

    # TODO: need to make sure we are not running on any facts we trained on, I think
    for i, fact_id in enumerate(fact_ids):
        logger.info(f"Processing fact_id: {fact_id}")

        probe_df, eval_df = dataset.get_splits(fact_id)
        logger.info(
            f"  Probe training: {len(probe_df)} rows, eval: {len(eval_df)} rows"
        )

        # Train two probe sets for this fact: real (honest labels) and
        # shuffled (labels permuted — N2 null control).
        probes_real, scores_real = train_probes(
            model, tokenizer, probe_df, shuffle_labels=False
        )
        probes_shuffled, scores_shuffled = train_probes(
            model, tokenizer, probe_df, shuffle_labels=True
        )
        probe_test_scores["real"][fact_id] = {
            str(k): v for k, v in scores_real.items()
        }
        probe_test_scores["shuffled"][fact_id] = {
            str(k): v for k, v in scores_shuffled.items()
        }

        probe_variants = [("real", probes_real), ("shuffled", probes_shuffled)]

        # Simulate conversations on held-out eval rows
        for _, row in eval_df.iterrows():
            logger.info(f"  Simulating conversation for pair_id {row['pair_id']}")

            conversation, per_turn_activations = simulate_conversation(
                model, tokenizer, partner, row
            )

            _emit_records(
                probe_variants,
                per_turn_activations[activation_source],
                conversation,
                fact_id=fact_id,
                pair_id=row["pair_id"],
                fact_text=row["fact"],
                is_control=False,
                source_fact_id=fact_id,
            )

        # N1 cross-fact control: apply this fact's probes to a conversation
        # about a *different* fact. Rotation picks the next fact in the list
        # so every fact contributes one control-source conversation overall.
        control_fact_id = fact_ids[(i + 1) % len(fact_ids)]
        _, control_eval_df = dataset.get_splits(control_fact_id)
        control_row = control_eval_df.iloc[0]
        logger.info(
            f"  N1 control: applying {fact_id} probes to "
            f"{control_fact_id}/{control_row['pair_id']}"
        )
        control_conv, control_acts = simulate_conversation(
            model, tokenizer, partner, control_row
        )
        # Only apply the real probes here — a shuffled probe on a control
        # conversation is a double-null that carries no additional information
        # beyond the (shuffled, is_control=False) bucket, which already
        # characterizes the null distribution.
        _emit_records(
            [("real", probes_real)],
            control_acts[activation_source],
            control_conv,
            fact_id=fact_id,
            pair_id=control_row["pair_id"],
            fact_text=control_row["fact"],
            is_control=True,
            source_fact_id=control_fact_id,
        )

    # --- Write final results ---
    final_results = {
        "metadata": {
            "model": model_checkpoint,
            "partner_model": partner_checkpoint,
            "num_turns": num_turns,
            "probe_activation_source": activation_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "probe_test_scores": probe_test_scores,
        },
        "turns": all_turns,
    }

    # TODO: analyze in a notebook first

    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Results written to {output_file}")
    return final_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        settings_file_path = sys.argv[1]
        init_settings(settings_file_path)
    else:
        logger.error("No settings file path provided.")
        sys.exit(1)

    run_benchmark()
