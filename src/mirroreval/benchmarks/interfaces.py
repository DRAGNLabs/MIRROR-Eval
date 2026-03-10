from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional, Dict, Type

DATASETS: Dict[str, Type["DatasetInterface"]] = {}
METRICS: Dict[str, Type["MetricInterface"]] = {}


def register_dataset(name: str):
    """Decorator to register a dataset class by name."""

    def decorator(cls: Type["DatasetInterface"]):
        DATASETS[name] = cls
        return cls

    return decorator


def register_metric(name: str):
    """Decorator to register a metric function by name."""

    def decorator(cls: Type["MetricInterface"]):
        METRICS[name] = cls
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


class MetricInterface(ABC):
    """Abstract base class for metrics."""

    @abstractmethod
    def __call__(self, dataset):
        pass
