from pathlib import Path
from enum import Enum

from mirroreval.creativity_development.creativity_entrypoint import (
    launch_creativity_evaluation,
)

from .config import init_settings, settings
from .logger import logger


class Benchmarks(str, Enum):
    CREATIVITY = "creativity"
    # Future benchmarks can be added here


BENCHMARK_SCRIPTS = {
    Benchmarks.CREATIVITY: launch_creativity_evaluation,
    # Future benchmarks can be mapped here
}


def evaluate(settings_file_path):
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
    logger.info("MIRROR-Eval: Initializing settings...")

    # Get the absolute path of the config file
    settings_file_path = Path(settings_file_path).resolve()

    init_settings(settings_file_path)

    # Iterate through benchmarks specified in settings
    for benchmark_name in settings.benchmarks.benchmarks:
        if benchmark_name not in BENCHMARK_SCRIPTS:
            logger.error(
                f"Warning: Unknown benchmark '{benchmark_name}' specified in settings."
            )
            continue
        benchmark = Benchmarks(benchmark_name)
        logger.info(f"Running benchmark: {benchmark.value}")
        BENCHMARK_SCRIPTS[benchmark]()
