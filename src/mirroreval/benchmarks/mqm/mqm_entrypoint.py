"""
MQM Entrypoint
==============
Entry function for the MT Metric Reliability (MQM) benchmark.

Responsibilities:
  1. Pre-download the model under test and FLORES+ dataset so that SLURM
     compute nodes (which run with HF_HUB_OFFLINE=1) can access them offline.
  2. Launch the benchmark locally or via SLURM.
"""

from mirroreval.config import settings
from mirroreval.benchmarks.mqm.mqm_benchmark import run_benchmark
from mirroreval.hf_utilities import download_from_hf, load_hf_dataset
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job
from mirroreval.logger import logger


def launch_mqm_evaluation():
    logger.info("MIRROR-Eval: MQM (MT Metric Reliability) benchmark starting...")

    # --- Pre-download: translation model under test ---
    logger.info(f"Caching translation model: {settings.model.model_checkpoint_path}")
    download_from_hf(settings.model.model_checkpoint_path)

    # --- Pre-download: FLORES+ dataset ---
    logger.info("Caching FLORES+ dataset...")
    load_hf_dataset("openlanguagedata/flores_plus")

    # --- Pre-download: COMET model weights (if COMET is in the metrics list) ---
    if "comet" in settings.mqm.metrics:
        comet_model = settings.mqm.get("comet_model", "Unbabel/wmt22-comet-da")
        logger.info(f"Caching COMET model: {comet_model}")
        try:
            from comet import download_model
            download_model(comet_model)
        except ImportError:
            logger.warning(
                "unbabel-comet not installed; COMET metric will be skipped. "
                "Install with: pip install unbabel-comet"
            )
        except Exception as e:
            logger.warning(f"Could not pre-download COMET model: {e}")

    # --- Launch ---
    if settings.slurm_job.use_slurm:
        logger.info("Submitting MQM benchmark to SLURM...")
        rendered_slurm_script = render_slurm_script(
            script_name="benchmarks/mqm/mqm_benchmark.py"
        )
        submit_slurm_job(rendered_slurm_script)
    else:
        run_benchmark()
