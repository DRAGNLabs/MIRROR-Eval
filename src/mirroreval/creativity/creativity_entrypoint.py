from huggingface_hub import login

from mirroreval.call_hf_model import (
    download_hf_model,
    download_tokenizer,
    download_dataset,
)
from mirroreval.config import settings
from mirroreval.slurm_utilities import render_slurm_script
from mirroreval.creativity.creativity_metric import run_metric


def launch_creativity_evaluation():
    print("MIRROR-Eval: Creativity evaluation starting...")

    # Ensure we're logged in to Hugging Face
    login()

    # Download models/data/tokenizers if necessary
    models = ["meta-llama/Llama-3.3-70B-Instruct", "google/gemma-7b", "Qwen/Qwen3-0.6B"]
    for model in models:
        download_hf_model(model)

    # Launch evaluation
    if settings.slurm_job.use_slurm is True:
        print("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(script_name="creativity_metric.py")
        print(rendered_slurm_script)
        # TODO: actually submit the job
    else:
        run_metric()
