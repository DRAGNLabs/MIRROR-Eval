"""
Creativity Datasets
===================
Defines dataset classes for the creativity benchmark.

Each dataset class implements DatasetInterface and is registered with
@register_dataset. The decorator associates the class with a string identifier. 
The registration name must exactly match the dataset string used in settings.toml. 
When creativity_benchmark.py imports this module, the decorator
fires and adds the class to the global DATASETS dict.

This creativity benchmark expects datasets with numbered multiturn
columns such as `P1`, `P2`, `R1`, `R2`, etc. The preprocessing pipeline then
extracts those columns into turn- or sentence-level rows before embedding.

To add a new dataset for this creativity benchmark, define a new class here with @register_dataset
and add the corresponding name to setting.toml under [creativity].datasets.
"""

from __future__ import annotations

from typing import Any, Iterator

from mirroreval.benchmarks.creativity.creativity_message_processing import load_multi_turn_dataset

from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset

# The string "jackwarner/multi-turn-conversations" must match exactly what appears in
# settings.toml's [creativity].datasets list. This is how the benchmark knows
# which class to instantiate for a given dataset name.
@register_dataset("jackwarner/multi-turn-conversations")
class MultiTurnConversationsDataset(DatasetInterface):
    """
    Dataset wrapper for the canonical MIRROR-CAP multiturn conversation data.

    The underlying data is loaded from Hugging Face and normalized so the
    downstream benchmark can rely on a stable schema, including a `row_id`
    column for traceability across exploded turn and sentence rows.
    """

    def __init__(self):
        self.dataset = None
        # load_data is called immediately on construction so the dataset
        # is ready to iterate as soon as the instance is created.
        self.load_data()

    def load_data(self) -> None:
        """Load and normalize the multiturn dataset from Hugging Face."""
        self.dataset = load_multi_turn_dataset("jackwarner/multi-turn-conversations")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Yield one raw training example at a time as a plain dictionary.

        The creativity benchmark usually consumes `self.dataset` directly for
        row-building, but iteration support is still provided to satisfy the
        DatasetInterface contract and to keep the dataset class usable in other
        contexts.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example
