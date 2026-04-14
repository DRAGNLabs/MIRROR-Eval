from pathlib import Path

from mirroreval.benchmarks.mta.mta_entrypoint import launch_mta_evaluation
from mirroreval.benchmarks.mqm.mqm_entrypoint import launch_mqm_evaluation

from .config import init_settings, settings
from .logger import logger

BENCHMARKS = {
    "mta": launch_mta_evaluation,
    "mqm": launch_mqm_evaluation,
}


def evaluate(settings_file_path):
    """
    Run the MIRROR evaluation pipeline.

    Args:
        settings_file_path: Path to a TOML settings file.

    Examples:
        >>> from mirroreval import evaluate
        >>> evaluate("settings.toml")
    """
    logger.info("MIRROR-Eval: Initializing settings...")

    # Get the absolute path of the config file
    settings_file_path = Path(settings_file_path).resolve()

    init_settings(settings_file_path)

    # Iterate through benchmarks specified in settings
    for name in settings.benchmarks.benchmarks:
        if name not in BENCHMARKS:
            logger.error(f"Unknown benchmark '{name}' specified in settings.")
            continue
        logger.info(f"Running benchmark: {name}")
        BENCHMARKS[name]()
