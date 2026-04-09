import json

import numpy as np
from datasets import Dataset, DatasetDict

from mirroreval.benchmarks.creativity.creativity_benchmark import run_benchmark
from mirroreval.benchmarks.creativity.creativity_simulate_conversation import (
    simulate_conversation,
)
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
        self.rows = [
            {
                "ID": 101,
                "row_id": 0,
                "archetypes": ["prompt a", "follow a", "follow aa"],
            },
            {
                "ID": 102,
                "row_id": 1,
                "archetypes": ["prompt b", "follow b", "follow bb"],
            },
        ]

    def __iter__(self):
        return iter(self.rows)


class FakePromptFollowupDataset:
    def __init__(self):
        self.rows = [
            {
                "ID": 201,
                "row_id": 0,
                "prompt": "starter a",
                "followup_1": "followup a1",
                "followup_2": "followup a2",
                "followup_3": "followup a3",
                "archetype_name": "example_archetype",
            },
            {
                "ID": 202,
                "row_id": 1,
                "prompt": "starter b",
                "followup_1": "followup b1",
                "followup_2": "followup b2",
                "followup_3": "followup b3",
                "archetype_name": "example_archetype",
            },
        ]

    def __iter__(self):
        return iter(self.rows)


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
    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_benchmark.simulate_conversation",
        lambda dataset, output_file=None: DatasetDict(
            {
                "train": Dataset.from_dict(
                    {
                        "ID": [101, 102],
                        "row_id": [0, 1],
                        "P1": ["prompt a", "prompt b"],
                        "P2": ["follow a", "follow b"],
                        "P3": ["follow aa", "follow bb"],
                        "R1": ["alpha. beta.", "one. two."],
                        "R2": ["alpha. gamma.", "one. three."],
                        "R3": ["delta.", "four."],
                    }
                )
            }
        ),
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
    model_responses_path = results_dir / "creativity_model_responses.jsonl"
    summary_path = results_dir / "creativity_summary.jsonl"

    assert model_responses_path.parent.exists()
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


def test_creativity_benchmark_skips_if_summary_exists(tmp_path, monkeypatch):
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

    pairwise_path = results_dir / "creativity_pairwise_results.jsonl"
    summary_path = results_dir / "creativity_summary.jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    pairwise_path.write_text("existing-pairwise\n", encoding="utf-8")
    summary_path.write_text("existing-summary\n", encoding="utf-8")

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("benchmark work should be skipped when summary exists")

    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_benchmark.TextEmbedder",
        _fail_if_called,
    )

    summary = run_benchmark()

    assert summary is None
    assert pairwise_path.read_text(encoding="utf-8") == "existing-pairwise\n"
    assert summary_path.read_text(encoding="utf-8") == "existing-summary\n"


def test_creativity_simulate_conversation_from_archetypes(tmp_path, monkeypatch):
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        """
[model]
model_checkpoint_path = "distilgpt2"

[slurm_job]
settings_file_path = ""
""".strip()
    )

    init_settings(config_path)

    class FakeTokenizer:
        model_max_length = 1024

        def encode(self, text):
            return list(range(len(text)))

        def decode(self, input_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "x" * len(input_ids)

    def fake_pipe(prompt, max_new_tokens, max_length):
        del max_new_tokens, max_length
        if isinstance(prompt, list):
            response_number = sum(1 for message in prompt if message["role"] == "user")
            return [
                {
                    "generated_text": [
                        *prompt,
                        {"role": "assistant", "content": f"reply {response_number}"},
                    ]
                }
            ]

        response_number = prompt.count("User:")
        return [{"generated_text": f"{prompt} reply {response_number}"}]

    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_simulate_conversation.get_hf_pipeline",
        lambda model_name: fake_pipe,
    )
    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_simulate_conversation.has_chat_template",
        lambda model_name: False,
    )
    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_simulate_conversation.get_hf_tokenizer",
        lambda model_name: FakeTokenizer(),
    )

    dataset = FakeDataset()
    generated = simulate_conversation(dataset)
    train = generated["train"]

    assert train["P1"] == ["prompt a", "prompt b"]
    assert train["P2"] == ["follow a", "follow b"]
    assert train["P3"] == ["follow aa", "follow bb"]
    assert train["R1"] == ["reply 1", "reply 1"]
    assert train["R2"] == ["reply 2", "reply 2"]
    assert train["R3"] == ["reply 3", "reply 3"]


def test_creativity_simulate_conversation_from_prompt_followups(tmp_path, monkeypatch):
    config_path = tmp_path / "settings.toml"
    config_path.write_text(
        """
[model]
model_checkpoint_path = "distilgpt2"

[slurm_job]
settings_file_path = ""
""".strip()
    )

    init_settings(config_path)

    class FakeTokenizer:
        model_max_length = 1024

        def encode(self, text):
            return list(range(len(text)))

        def decode(self, input_ids, skip_special_tokens=True):
            del skip_special_tokens
            return "x" * len(input_ids)

    def fake_pipe(prompt, max_new_tokens, max_length):
        del max_new_tokens, max_length
        response_number = prompt.count("User:")
        return [{"generated_text": f"{prompt} reply {response_number}"}]

    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_simulate_conversation.get_hf_pipeline",
        lambda model_name: fake_pipe,
    )
    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_simulate_conversation.has_chat_template",
        lambda model_name: False,
    )
    monkeypatch.setattr(
        "mirroreval.benchmarks.creativity.creativity_simulate_conversation.get_hf_tokenizer",
        lambda model_name: FakeTokenizer(),
    )

    output_file = tmp_path / "creativity_model_responses.jsonl"
    dataset = FakePromptFollowupDataset()
    generated = simulate_conversation(dataset, output_file=output_file)
    train = generated["train"]

    assert train["P1"] == ["starter a", "starter b"]
    assert train["P2"] == ["followup a1", "followup b1"]
    assert train["P3"] == ["followup a2", "followup b2"]
    assert train["P4"] == ["followup a3", "followup b3"]
    assert train["R1"] == ["reply 1", "reply 1"]
    assert train["R2"] == ["reply 2", "reply 2"]
    assert train["R3"] == ["reply 3", "reply 3"]
    assert train["R4"] == ["reply 4", "reply 4"]

    records = [
        json.loads(line) for line in output_file.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["prompt"] == "starter a"
    assert records[0]["followup_1"] == "followup a1"
    assert records[0]["followup_2"] == "followup a2"
    assert records[0]["followup_3"] == "followup a3"
    assert records[0]["response_1"] == "reply 1"
    assert records[0]["response_2"] == "reply 2"
    assert records[0]["response_3"] == "reply 3"
    assert records[0]["response_4"] == "reply 4"
