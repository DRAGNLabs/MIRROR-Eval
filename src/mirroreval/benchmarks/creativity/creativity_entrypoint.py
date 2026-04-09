"""
Creativity Entrypoint
=====================
This is the launch script for the embedding-based creativity benchmark.

Every benchmark in MIRROR-Eval needs an entrypoint like this one. 
It has two purposes:
  1. Pre-download any models and datasets so they are available in the local
     Hugging Face cache before execution begins.
  2. Launch the benchmark either locally or through SLURM, depending on the
     user configuration in settings.toml.

The benchmark loads archetypal prompt turns, simulates model responses,
embeds the selected conversation turns, and computes pairwise
persistence/novelty/creativity metrics.
"""

from __future__ import annotations

from mirroreval.benchmarks.creativity.creativity_benchmark import run_benchmark
from mirroreval.benchmarks.creativity.creativity_embedding_model import resolve_embedding_model_name
from mirroreval.config import settings
from mirroreval.hf_utilities import download_from_hf, load_hf_dataset
from mirroreval.logger import logger
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job


def launch_creativity_evaluation():
    """
    Start the creativity benchmark.

    This function mirrors the framework contract used by other benchmarks:
      - resolve and pre-download the embedding model configured for the run
      - pre-download every configured dataset used as prompt/archetype input
      - either submit a SLURM job or execute the benchmark locally

    The function is registered in evaluate.py's BENCHMARKS dict under the
    name "creativity".
    """
    logger.info("MIRROR-Eval: Creativity evaluation starting...")

    embedding_model = resolve_embedding_model_name(settings.creativity.embedding_model)
    # --- Step 1: Pre-download artifacts ---
    # download_from_hf caches the model via huggingface_hub.snapshot_download.
    # This ensures the model weights are available offline when SLURM jobs run.
    logger.info(f"Ensuring embedding model is downloaded: {embedding_model}")
    download_from_hf(embedding_model)

    # Also pre-download every dataset listed in the config. Like models, these
    # need to be cached before compute nodes try to access them.
    for dataset in settings.creativity.datasets:
        load_hf_dataset(dataset)

    # --- Step 2: Launch the benchmark ---
    # The config's slurm_job.use_slurm flag determines the execution path.
    # - SLURM: render a job script from the Jinja2 template and submit it.
    #   The script will invoke mta_benchmark.py as a standalone process.
    # - Local: call run_benchmark() directly in the current process.
    if settings.slurm_job.use_slurm:
        logger.info("Submitting creativity job to SLURM...")
        rendered_slurm_script = render_slurm_script(
            script_name="benchmarks/creativity/creativity_benchmark.py"
        )
        submit_slurm_job(rendered_slurm_script)
    else:
        run_benchmark()
