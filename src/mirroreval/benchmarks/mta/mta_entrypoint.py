from mirroreval.config import settings
from mirroreval.benchmarks.mta.mta_benchmark import run_benchmark
from mirroreval.hf_utilities import download_from_hf, load_hf_dataset
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job
from mirroreval.logger import logger


def launch_mta_evaluation():
    logger.info("MIRROR-Eval: MTA evaluation starting...")

    # Download models/data/tokenizers if necessary
    logger.info(f"Ensuring model is downloaded: {settings.mta.llm_judge_model}")
    download_from_hf(settings.mta.llm_judge_model)

    for dataset in settings.mta.datasets:
        load_hf_dataset(dataset)

    # Launch evaluation
    if settings.slurm_job.use_slurm:
        logger.info("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(
            script_name="benchmarks/mta/mta_benchmark.py"
        )
        submit_slurm_job(rendered_slurm_script)
    else:
        run_benchmark()
