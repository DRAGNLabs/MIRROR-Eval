from typing import Any, Iterator

from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset
from mirroreval.hf_utilities import load_hf_dataset


@register_dataset("royal42/mta-test")
class MTA_test(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("royal42/mta-test")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example
