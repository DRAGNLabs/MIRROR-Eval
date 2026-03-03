from mirroreval.config import settings
from mirroreval.benchmarks.creativity.creativity_benchmark import run_benchmark
from mirroreval.hf_utilities import download_from_hf, load_hf_dataset
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job
from mirroreval.logger import logger


def launch_creativity_evaluation():
    logger.info("MIRROR-Eval: Creativity evaluation starting...")

    # Download models/data/tokenizers if necessary
    for model in settings.creativity.llm_judge_models:
        download_from_hf(model)

    for model in settings.creativity.other_models:
        download_from_hf(model)

    for dataset in settings.creativity.datasets:
        # If the dataset ends with "demo", skip downloading
        if dataset.endswith("demo"):
            logger.info(f"Skipping download for demo dataset: {dataset}")
            continue
        load_hf_dataset(dataset)

    # Launch evaluation
    if settings.slurm_job.use_slurm:
        logger.info("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(
            script_name="benchmarks/creativity/creativity_benchmark.py"
        )
        submit_slurm_job(rendered_slurm_script)
    else:
        run_benchmark()
