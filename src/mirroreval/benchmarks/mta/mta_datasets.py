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

import pandas as pd

from typing import Any, Iterator, Tuple

from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset
from mirroreval.hf_utilities import load_hf_dataset


@register_dataset("royal42/mta")
class MTA(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        raw = load_hf_dataset("royal42/mta")
        df = raw["train"].to_pandas()
        assert isinstance(df, pd.DataFrame)
        self.dataset = df

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for _, row in self.dataset.iterrows():
            yield row.to_dict()

    def iter_fact_id(self) -> Iterator[str]:
        """Yield each unique fact_id in the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        yield from self.dataset["fact_id"].unique()

    def get_splits(
        self,
        fact_id: str,
        probe_frac: float = 0.6,
        n_eval: int = 5,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split rows for a given fact_id into probe-training and eval sets.

        Args:
            fact_id: Which fact to filter to.
            probe_frac: Fraction of rows allocated to probe training.
            n_eval: Number of rows to sample from the remainder for evaluation.
            seed: Random seed for reproducibility.

        Returns:
            (probe_df, eval_df) with non-overlapping indices.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        fact_rows = self.dataset[self.dataset["fact_id"] == fact_id]
        shuffled = fact_rows.sample(frac=1.0, random_state=seed)

        split_idx = int(len(shuffled) * probe_frac)
        probe_df = shuffled.iloc[:split_idx]
        remainder = shuffled.iloc[split_idx:]

        n_eval = min(n_eval, len(remainder))
        eval_df = remainder.head(n_eval)

        return probe_df, eval_df
