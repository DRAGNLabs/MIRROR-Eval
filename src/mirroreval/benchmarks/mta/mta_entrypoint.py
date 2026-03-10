"""
MTA Entrypoint
==============
This is the entry script for the Multi-Turn Accuracy (MTA) benchmark.

Every benchmark in MIRROR-Eval must have an entrypoint like this one. It serves
two purposes:
  1. Pre-download any models and datasets so they are cached locally. This is
     required because compute nodes on HPC clusters often lack internet access.
  2. Launch the benchmark — either locally or by submitting a SLURM job.

The launch function defined here is registered in evaluate.py's BENCHMARKS dict,
which is how the framework discovers and invokes it.
"""

from mirroreval.config import settings
from mirroreval.benchmarks.mta.mta_benchmark import run_benchmark
from mirroreval.hf_utilities import download_from_hf, load_hf_dataset
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job
from mirroreval.logger import logger


def launch_mta_evaluation():
    logger.info("MIRROR-Eval: MTA evaluation starting...")

    # --- Step 1: Pre-download artifacts ---
    # download_from_hf caches the model via huggingface_hub.snapshot_download.
    # This ensures the model weights are available offline when SLURM jobs run.
    logger.info(f"Ensuring model is downloaded: {settings.mta.llm_judge_model}")
    download_from_hf(settings.mta.llm_judge_model)

    # Also pre-download every dataset listed in the config. Like models, these
    # need to be cached before compute nodes try to access them.
    for dataset in settings.mta.datasets:
        load_hf_dataset(dataset)

    # --- Step 2: Launch the benchmark ---
    # The config's slurm_job.use_slurm flag determines the execution path.
    # - SLURM: render a job script from the Jinja2 template and submit it.
    #   The script will invoke mta_benchmark.py as a standalone process.
    # - Local: call run_benchmark() directly in the current process.
    if settings.slurm_job.use_slurm:
        logger.info("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(
            script_name="benchmarks/mta/mta_benchmark.py"
        )
        submit_slurm_job(rendered_slurm_script)
    else:
        run_benchmark()
