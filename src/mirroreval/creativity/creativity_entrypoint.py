from huggingface_hub import login

from mirroreval.call_hf_model import call_hf_model
from mirroreval.config import settings
from mirroreval.slurm_utilities import render_slurm_script
from mirroreval.creativity.creativity_metric import run_metric


def launch_creativity_evaluation():
    print("MIRROR-Eval: Creativity evaluation starting...")

    # Ensure we're logged in to Hugging Face
    login()

    # Download models/data/tokenizers if necessary

    # Launch evaluation
    if settings.slurm_job.use_slurm is True:
        print("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(script_name="creativity_metric.py")
        print(rendered_slurm_script)
    else:
        run_metric()
