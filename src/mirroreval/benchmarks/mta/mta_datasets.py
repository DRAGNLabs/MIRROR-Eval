"""
MTA Datasets
=============
Defines dataset classes for the MTA benchmark.

Each class implements DatasetInterface and is registered with @register_dataset.
The decorator associates the class with a string identifier (matching the
dataset name in settings.toml). When mta_benchmark.py imports this module,
the decorator fires and adds the class to the global DATASETS dict.

To add a new dataset for MTA, define a new class here with @register_dataset
and add the corresponding name to settings.toml under [mta].datasets.
"""

from typing import Any, Iterator

from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset
from mirroreval.hf_utilities import load_hf_dataset


# The string "royal42/mta-test" must match exactly what appears in
# settings.toml's [mta].datasets list. This is how the benchmark knows
# which class to instantiate for a given dataset name.
@register_dataset("royal42/mta-test")
class MTA_test(DatasetInterface):
    def __init__(self):
        self.dataset = None
        # load_data is called immediately on construction so the dataset
        # is ready to iterate as soon as the instance is created.
        self.load_data()

    def load_data(self) -> None:
        # Loads the dataset from HuggingFace Hub. The data is cached locally
        # after the first download (which is handled by the entrypoint).
        self.dataset = load_hf_dataset("royal42/mta-test")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Yield one example at a time as a plain dict.

        Each example contains:
          - "prompt": the initial user message
          - "followup_1", "followup_2", "followup_3": subsequent user messages

        Batching is NOT handled here — that is the responsibility of whatever
        consumes the dataset (e.g., simulate_conversation or a metric).
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example
