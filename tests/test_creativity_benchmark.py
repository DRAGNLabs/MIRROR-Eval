import json

import numpy as np
from datasets import Dataset, DatasetDict

from mirroreval.benchmarks.creativity.creativity_benchmark import run_benchmark
from mirroreval.config import init_settings


class FakeEmbedder:
    def __init__(self, model_name):
        self.model_name = model_name

    def embed(
        self,
        texts,
        *,
        batch_size,
        max_length,
        normalize,
        show_progress,
    ):
        del batch_size, max_length, normalize, show_progress
        vectors = []
        for text in texts:
            value = float(len(text))
            vectors.append(np.array([value, value / 2.0], dtype=np.float32))
        return np.vstack(vectors)


class FakeDataset:
    def __init__(self):
        self.dataset = DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "ID": [101, 102],
                        "row_id": [0, 1],
                        "R1": ["alpha. beta.", "one. two."],
                        "R2": ["alpha. gamma.", "one. three."],
                        "R3": ["delta.", "four."],
                        "P1": ["prompt a", "prompt b"],
                        "P2": ["follow a", "follow b"],
                        "P3": ["follow aa", "follow bb"],
                    }
                )
            }
        )


def test_creativity_benchmark_generates_outputs(tmp_path, monkeypatch):
    config_path = tmp_path / "settings.toml"
    results_dir = tmp_path / "results"
    config_path.write_text(
        f"""
[model]
model_checkpoint_path = "distilgpt2"

[benchmarks]
benchmarks = ["creativity"]

[slurm_job]
use_slurm = false
job_name = "test"
time = "00:10:00"
ntasks = 1
nodes = 1
mem_per_cpu = "1G"
gpus = 0
conda_env = "mirror-eval"
qos = ""

[creativity]
metrics = ["embedding-creativity"]
datasets = ["jackwarner/multi-turn-conversations"]
role = "assistant"
mode = "sentence"
pair_mode = "all"
threshold = 0.85
embedding_model = "minilm"
batch_size = 8
max_length = 64
normalize_embeddings = true
max_items = 64
output_dir = "{results_dir}"
""".strip()
    )

    init_settings(config_path)

    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_benchmark.TextEmbedder",
        FakeEmbedder,
    )
    monkeypatch.setitem(
        __import__(
            "mirroreval.benchmarks.creativity.creativity_benchmark",
            fromlist=["DATASETS"],
        ).DATASETS,
        "jackwarner/multi-turn-conversations",
        FakeDataset,
    )

    summary = run_benchmark()

    pairwise_path = results_dir / "creativity_pairwise_results.jsonl"
    summary_path = results_dir / "creativity_summary.jsonl"

    assert pairwise_path.exists()
    assert summary_path.exists()
    assert summary["total_pairs"] > 0

    pairwise_records = [
        json.loads(line)
        for line in pairwise_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        record["metric_name"] == "embedding-creativity"
        for record in pairwise_records
    )
    assert all(
        record["dataset_name"] == "jackwarner/multi-turn-conversations"
        for record in pairwise_records
    )
