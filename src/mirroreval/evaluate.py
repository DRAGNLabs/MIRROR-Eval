from .config import settings, init_settings
import argparse
from .creativity import creativity
from .slurm_utilities import render_slurm_script


def evaluate(config_path: str):
    """
    Run the MIRROR evaluation pipeline.

    This is the primary entrypoint for the MIRROR-Eval package. It will
    orchestrate the evaluation pipeline for MIRROR models.

    Args:
        *args: Positional arguments for the evaluation pipeline.
        **kwargs: Keyword arguments for the evaluation pipeline.

    Returns:
        dict: Evaluation results containing metrics and outputs.

    Examples:
        >>> from mirror_eval import evaluate
        >>> results = evaluate()
        >>> print(results)

    Note:
        This is a placeholder implementation. The full evaluation pipeline
        will be implemented in future versions.
    """
    print("MIRROR-Eval: Evaluation pipeline starting...")

    init_settings(config_path)

    print(settings.as_dict())
    # Run creativity evaluation
    if settings.slurm_job.use_slurm is True:
        print("Submitting job to SLURM...")
        rendered_slurm_script = render_slurm_script(script_name="creativity.py")
        print(rendered_slurm_script)
    else:
        creativity()
