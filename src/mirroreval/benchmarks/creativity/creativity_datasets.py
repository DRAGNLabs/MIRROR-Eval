"""
Creativity Datasets
===================
Defines dataset classes for the creativity benchmark.

Each dataset class implements DatasetInterface and is registered with
@register_dataset. The decorator associates the class with a string identifier. 
The registration name must exactly match the dataset string used in settings.toml. 
When creativity_benchmark.py imports this module, the decorator
fires and adds the class to the global DATASETS dict.

This creativity benchmark accepts source datasets that can describe ordered
user-turn archetypes. The conversation simulator normalizes those inputs into
MTA-style user turns, generates model responses, and then materializes the
`P*`/`R*` columns consumed by the embedding pipeline.

To add a new dataset for this creativity benchmark, define a new class here with @register_dataset
and add the corresponding name to setting.toml under [creativity].datasets.
"""

from __future__ import annotations

from typing import Any, Iterator

from mirroreval.benchmarks.creativity.creativity_message_processing import (
    load_multi_turn_dataset,
)

from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset
from mirroreval.hf_utilities import load_hf_dataset

# The string "jackwarner/multi-turn-conversations" must match exactly what appears in
# settings.toml's [creativity].datasets list. This is how the benchmark knows
# which class to instantiate for a given dataset name.
@register_dataset("jackwarner/multi-turn-conversations")
class MultiTurnConversationsDataset(DatasetInterface):
    """
    Dataset wrapper for the canonical MIRROR-CAP multiturn conversation data.

    The underlying data is loaded from Hugging Face and normalized so the
    simulator can recover ordered prompt turns and preserve a stable `row_id`
    for downstream traceability.
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

        Each example contains:
          - "prompt": the initial user message
          - "followup_1", "followup_2", "followup_3": subsequent user messages

        Batching is NOT handled here — that is the responsibility of whatever
        consumes the dataset (e.g., simulate_conversation or a metric).

        The creativity benchmark usually consumes `self.dataset` directly for
        row-building, but iteration support is still provided to satisfy the
        DatasetInterface contract and to keep the dataset class usable in other
        contexts.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example


@register_dataset("jackwarner/creativity-test")
class CreativityTestDataset(DatasetInterface):
    """
    Dataset wrapper for prompt-only MTA-style creativity inputs.

    The dataset is expected to provide:
      - `prompt`
      - `followup_1`, `followup_2`, `followup_3`
    Additional metadata columns such as `archetype_name` are preserved but
    ignored by the simulator.
    """

    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        """Load the prompt/followup creativity test dataset from Hugging Face."""
        self.dataset = load_hf_dataset("jackwarner/creativity-test")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example


@register_dataset("jackwarner/creativity-smoke-test")
class CreativitySmokeTestDataset(DatasetInterface):
    """Dataset wrapper for a tiny prompt-only creativity smoke dataset."""

    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("jackwarner/creativity-smoke-test")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example
