import subprocess

from mirroreval.hf_utilities import (
    download_hf_model,
    download_tokenizer,
    download_hf_dataset,
)
from mirroreval.config import settings
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job
from mirroreval.creativity.creativity_metric import run_metric


def launch_creativity_evaluation():
    print("MIRROR-Eval: Creativity evaluation starting...")

    # Download models/data/tokenizers if necessary
    models = settings.creativity.models

    dataset = "royal42/gcr-diversity"

    for model in models:
        download_hf_model(model)

    download_hf_dataset(dataset)

    # Launch evaluation
    if settings.slurm_job.use_slurm is True:
        print("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(
            script_name="creativity/creativity_metric.py"
        )
        print(rendered_slurm_script)
        submit_slurm_job(rendered_slurm_script)
    else:
        run_metric()
