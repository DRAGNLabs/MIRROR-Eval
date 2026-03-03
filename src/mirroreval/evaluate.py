from pathlib import Path

from mirroreval.benchmarks.creativity.creativity_entrypoint import (
    launch_creativity_evaluation,
)

from .config import init_settings, settings
from .logger import logger


BENCHMARKS = {
    "creativity": launch_creativity_evaluation,
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
            logger.error(
                f"Unknown benchmark '{name}' specified in settings."
            )
            continue
        logger.info(f"Running benchmark: {name}")
        BENCHMARKS[name]()
