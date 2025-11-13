from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Dict, Type
from dataclasses import dataclass

from mirroreval.hf_utilities import load_hf_dataset

DATASETS: Dict[str, Type["DatasetInterface"]] = {}

# TODO: add way to retrieve list of datasets/models/metrics/benchmarks, etc. for settings


def register_dataset(name: str):
    """Decorator to register a dataset class by name."""

    def decorator(cls: Type["DatasetInterface"]):
        DATASETS[name] = cls
        return cls

    return decorator


class DatasetInterface(ABC):
    """Abstract base class for datasets."""

    @abstractmethod
    def load_data(self) -> None:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        pass

    def __len__(self) -> int:
        """Return dataset size, if applicable."""
        raise NotImplementedError("Length not supported for this dataset type.")

    def get_split(self, name: str) -> Optional["DatasetInterface"]:
        """Return a subset or split of the dataset, if applicable."""
        raise NotImplementedError("Splits not supported for this dataset type.")


@register_dataset("royal42/gcr-diversity")
class GCRDiversityDataset(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("royal42/gcr-diversity")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for split_name, split_dataset in self.dataset.items():
            for example in split_dataset:
                example["split"] = split_name

                # Add a field for the correct answer
                diversity_set_1 = example.get("Diversity_Set1")
                diversity_set_2 = example.get("Diversity_Set2")
                if diversity_set_1 > diversity_set_2:
                    example["correct_answer"] = 0
                elif diversity_set_2 > diversity_set_1:
                    example["correct_answer"] = 1
                else:
                    example["correct_answer"] = -1

                yield example


@register_dataset("royal42/gcr-diversity")
class GCRDiversityDemoDataset(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("royal42/gcr-diversity")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for split_name, split_dataset in self.dataset.items():
            for example in split_dataset.select(range(2)):
                example["split"] = split_name

                # Add a field for the correct answer
                diversity_set_1 = example.get("Diversity_Set1")
                diversity_set_2 = example.get("Diversity_Set2")
                if diversity_set_1 > diversity_set_2:
                    example["correct_answer"] = 0
                elif diversity_set_2 > diversity_set_1:
                    example["correct_answer"] = 1
                else:
                    example["correct_answer"] = -1

                yield example


@register_dataset("royal42/mt-eval-creativity")
class MTEvalCreativityDataset(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("royal42/mt-eval-creativity")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"]:
            yield example


@register_dataset("royal42/mt-eval-creativity-demo")
class MTEvalCreativityDemoDataset(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("royal42/mt-eval-creativity")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        for example in self.dataset["train"].select(range(2)):
            yield example
