# Contributing to MIRROR-Eval

This guide walks you through adding a new benchmark to the MIRROR-Eval pipeline.

It is expected that the benchmark being contributed has already been developed in a separate codebase. Developing a benchmark within MIRROR-Eval directly is not recommended.

The [MTA benchmark](../src/mirroreval/benchmarks/mta/) is a complete reference implementation. Review it alongside this guide.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [How the Pipeline Works](#how-the-pipeline-works)
3. [Contribution Requirements](#contribution-requirements)
4. [Utilities](#utilities)

---

## Getting Started

### Create a Branch

```bash
# Clone the repository
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

# Create a feature branch for your benchmark
git checkout -b add-<your-benchmark-name>
```

### Set Up for Development

**Using conda:**

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate mirror-eval

# Install in development mode
pip install -e ".[dev]"
```

### Verify the Installation

Run [demo.py](../demo.py) to confirm everything works:

```bash
python demo.py
```

`demo.py` is minimal — it just passes a config file to the evaluation entrypoint:

```python
import mirroreval

print(mirroreval.__version__)

config_path = "./settings.toml"

mirroreval.evaluate(config_path)
```

The configuration file ([settings.toml](../settings.toml)) is central to MIRROR-Eval. It specifies:

- **`model_checkpoint_path`** — the HuggingFace model to evaluate (default: `Qwen/Qwen3-0.6B`)
- **`benchmarks`** — which benchmarks to run
- **`slurm_job`** — parameters for running on a compute cluster
- **Benchmark-specific sections** — datasets, output directories, hyperparameters, etc.

Ensure the pipeline runs successfully with `demo.py` before continuing.

---

## How the Pipeline Works

When you run `demo.py` with the default configuration, MIRROR-Eval does the following:

```
settings.toml
    │
    ▼
evaluate()                          # Loads config, iterates benchmarks
    │
    ▼
launch_<benchmark>_evaluation()     # Entrypoint: downloads data, launches job
    │
    ├─ Downloads models/datasets    # Required — compute nodes may lack internet
    │
    ├─ Local or SLURM execution     # Determined by settings.slurm_job.use_slurm
    │
    ▼
run_benchmark()                     # Core benchmark logic
    │
    ├─ Load datasets                # Via registered dataset classes
    ├─ Generate model outputs       # Run the model under test
    ├─ Run metrics                  # Via registered metric classes
    ├─ Perform analysis             # Aggregate scores, compute statistics
    │
    ▼
Results JSON                        # Saved to output directory
```

In more detail:

1. `evaluate()` reads which benchmarks are listed in the config and calls each one's launch function.
2. The **entrypoint** downloads any necessary models or datasets (compute nodes on HPC systems often lack internet), then either submits a SLURM job or runs the benchmark locally.
3. The **benchmark** runs its evaluation logic — loading data, generating model outputs, scoring with metrics, and computing analysis.
4. The benchmark saves a results JSON to the configured output directory.

---

## Contribution Requirements

To add a benchmark, complete these six steps:

1. [Set up the file structure](#1-set-up-the-file-structure)
2. [Add benchmark logic](#2-add-benchmark-logic)
3. [Add an entrypoint script](#3-add-an-entrypoint-script)
4. [Update settings.toml](#4-update-settingstoml)
5. [Register your benchmark in evaluate.py](#5-register-your-benchmark-in-evaluatepy)
6. [Write benchmark documentation](#6-write-benchmark-documentation)

---

### 1. Set Up the File Structure

Create a directory for your benchmark inside [`src/mirroreval/benchmarks/`](../src/mirroreval/benchmarks/):

```
src/mirroreval/benchmarks/
└── your_benchmark/
    ├── __init__.py              # Empty, makes it a package
    ├── your_entrypoint.py       # Entrypoint (downloads + launches)
    ├── your_benchmark.py        # Core benchmark logic
    ├── your_datasets.py         # Dataset class(es) with @register_dataset
    ├── your_metrics.py          # Metric class(es) with @register_metric
    └── prompts.py               # Prompt templates (if needed)
```

The file names are up to you — this is just a recommended structure based on MTA.

---

### 2. Add Benchmark Logic

Your benchmark directory will contain the bulk of your code. You're free to organize it however you like, but there are three hard requirements:

1. **Load the model from config.** Read `settings.model.model_checkpoint_path` and load this model. This is the model being evaluated — your benchmark is responsible for loading it.

2. **Return and save results.** Your benchmark must return a results JSON with scores and save it to the output directory specified in the config.

3. **Provide an entrypoint.** See [step 3](#3-add-an-entrypoint-script).

Review the [MTA benchmark](../src/mirroreval/benchmarks/mta/) to see how these requirements are met in practice. The general flow in `run_benchmark()` is:

```python
def run_benchmark():
    # 1. Read settings
    metrics = settings.your_benchmark.metrics
    datasets = settings.your_benchmark.datasets

    # 2. Create output directory
    output_dir = Path(settings.your_benchmark.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load and process each dataset
    for dataset_name in datasets:
        dataset = DATASETS[dataset_name]()
        # ... run model, save intermediate results ...

    # 4. Run metrics
    for metric_name in metrics:
        METRICS[metric_name]()(intermediate_results_file)

    # 5. Compute final analysis and save
    final_scores = compute_scores(intermediate_results_file, output_file)
    return final_scores
```

---

### 3. Add an Entrypoint Script

Each benchmark must have an entrypoint function with two responsibilities:

1. **Download all necessary data and models** before the benchmark runs. This is required because compute nodes on HPC systems often lack internet access.
2. **Launch the benchmark** either locally or through SLURM, as specified in the config.

Copy [mta_entrypoint.py](../src/mirroreval/benchmarks/mta/mta_entrypoint.py) and modify it for your benchmark:

```python
from mirroreval.config import settings
from mirroreval.benchmarks.your_benchmark.your_benchmark import run_benchmark
from mirroreval.hf_utilities import download_from_hf, load_hf_dataset
from mirroreval.slurm_utilities import render_slurm_script, submit_slurm_job
from mirroreval.logger import logger


def launch_your_benchmark_evaluation():
    logger.info("MIRROR-Eval: Your benchmark starting...")

    # Download any models or datasets needed
    download_from_hf(settings.model.model_checkpoint_path)
    for dataset in settings.your_benchmark.datasets:
        load_hf_dataset(dataset)

    # Launch: SLURM or local
    if settings.slurm_job.use_slurm:
        rendered_slurm_script = render_slurm_script(
            script_name="benchmarks/your_benchmark/your_benchmark.py"
        )
        submit_slurm_job(rendered_slurm_script)
    else:
        run_benchmark()
```

---

### 4. Update settings.toml

Add a section for your benchmark in [settings.toml](../settings.toml). You can define any parameters your benchmark needs — datasets, models, hyperparameters, output paths, etc.

```toml
# Add your benchmark name to the list
[benchmarks]
benchmarks = [
    "mta",
    "your_benchmark"
]

# Add a section for your benchmark's parameters
[your_benchmark]
metrics = ["your-metric"]
datasets = ["your-org/your-dataset"]
output_dir = ""
# ... any other parameters your benchmark needs
```

Access these from your code via the global settings object:

```python
from mirroreval.config import settings

output_dir = Path(settings.your_benchmark.output_dir)
datasets = settings.your_benchmark.datasets
```

---

### 5. Register Your Benchmark in evaluate.py

Add your benchmark's launch function to the `BENCHMARKS` dict in [`evaluate.py`](../src/mirroreval/evaluate.py):

```python
from mirroreval.benchmarks.your_benchmark.your_entrypoint import launch_your_benchmark_evaluation

BENCHMARKS = {
    "mta": launch_mta_evaluation,
    "your_benchmark": launch_your_benchmark_evaluation,
}
```

The key in this dict must match the name used in `settings.toml`'s `benchmarks` list.

---

### 6. Write Benchmark Documentation

Create a markdown file in [`docs/benchmarks/`](../docs/benchmarks/) describing your benchmark. Copy [example.md](../docs/benchmarks/example.md) and fill out the sections. See [multiturn-accuracy.md](../docs/benchmarks/multiturn-accuracy.md) for a completed example.

Your documentation should cover:

- **About** — what the benchmark measures and how
- **Requirements** — any specific packages, data, or hardware needed
- **Inputs** — configurable settings in `settings.toml`
- **Outputs** — the structure and meaning of the results JSON

---

## Utilities

MIRROR-Eval provides several utilities to help structure your benchmark. These are used by the MTA benchmark and serve as good examples.

### Datasets

An abstract dataset interface is provided at [`benchmarks/interfaces.py`](../src/mirroreval/benchmarks/interfaces.py). Implement this interface to create a dataset class that the framework can discover and instantiate.

**The interface:**

```python
class DatasetInterface(ABC):

    @abstractmethod
    def load_data(self) -> None:
        """Load data from source (HuggingFace, local files, etc.)."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield one data point at a time as a dict."""
        pass

    def __len__(self) -> int:
        """Optional: return dataset size."""
        raise NotImplementedError

    def get_split(self, name: str) -> Optional["DatasetInterface"]:
        """Optional: return a named split (train, test, etc.)."""
        raise NotImplementedError
```

**Registering a dataset:**

Use the `@register_dataset` decorator to associate your class with a string identifier. This identifier should match the dataset name used in `settings.toml`.

```python
from mirroreval.benchmarks.interfaces import DatasetInterface, register_dataset
from mirroreval.hf_utilities import load_hf_dataset

@register_dataset("your-org/your-dataset")
class YourDataset(DatasetInterface):
    def __init__(self):
        self.dataset = None
        self.load_data()

    def load_data(self) -> None:
        self.dataset = load_hf_dataset("your-org/your-dataset")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        for example in self.dataset["train"]:
            yield example
```

**How registration works:** The decorator adds your class to a global `DATASETS` dict. When your benchmark imports the datasets module (as a side-effect import), the decorator fires and the class becomes available via `DATASETS["your-org/your-dataset"]()`. This is why MTA has this line:

```python
import mirroreval.benchmarks.mta.mta_datasets  # noqa: F401
```

The import looks unused, but it triggers the `@register_dataset` decorator.

**Design notes:**
- The dataset yields one example at a time via `__iter__`. Batching is the responsibility of metrics or benchmark logic.
- Each example is a plain `dict` — structure it however your benchmark needs.

---

### Metrics

An abstract metric interface is also provided in [`benchmarks/interfaces.py`](../src/mirroreval/benchmarks/interfaces.py). Metrics are callable objects that process benchmark outputs and produce scores.

**The interface:**

```python
class MetricInterface(ABC):

    @abstractmethod
    def __call__(self, dataset):
        """Run the metric on the given data and return/save scores."""
        pass
```

**Registering a metric:**

Use the `@register_metric` decorator, just like datasets:

```python
from mirroreval.benchmarks.interfaces import MetricInterface, register_metric

@register_metric("your-metric-name")
class YourMetric(MetricInterface):
    def __call__(self, dataset):
        # Process the data and compute scores
        # `dataset` can be a file path, generator, or any data source
        # — the convention is up to your benchmark
        pass
```

The string passed to `@register_metric` must match what's listed in `settings.toml` under your benchmark's `metrics` field. Like datasets, the class is registered via side-effect import:

```python
import mirroreval.benchmarks.your_benchmark.your_metrics  # noqa: F401
```

**MTA's LLM-as-a-Judge as an example:** In MTA, the metric receives a JSONL file path. It reads each line, batches them, runs an LLM judge pipeline, parses the JSON output for scores, and writes updated records back to the file atomically (via a temp file + `os.replace`). Your metric can follow this pattern or use a different approach.

---

### Prompts

If your benchmark uses LLM prompting (e.g., for LLM-as-a-judge scoring or conversation simulation), consider organizing prompts in a dedicated `prompts.py` module. MTA's [prompts.py](../src/mirroreval/benchmarks/mta/prompts.py) demonstrates this pattern.

**The pattern:**

```python
import copy

# Define prompt templates with format placeholders
PROMPTS = {
    "template_a": "Evaluate whether {response} addresses {prompt}. Return JSON: ...",
    "template_b": "Is {response} relevant to {prompt}? Yes/No. Return JSON: ...",
}

# Define model-specific conversation formats
SYSTEM_PROMPTS = {
    "model-org/model-name": [
        {"role": "system", "content": "You are a helpful evaluator."},
        {"role": "user", "content": "{prompt}"},
    ],
}

def get_formatted_prompt(model_name, prompt_name, **kwargs):
    """Build a formatted conversation-style prompt."""
    prompt_template = copy.deepcopy(SYSTEM_PROMPTS[model_name])  # Don't mutate originals
    actual_prompt = PROMPTS[prompt_name].format(**kwargs)
    for message in prompt_template:
        if message["role"] == "user":
            message["content"] = message["content"].format(prompt=actual_prompt)
    return prompt_template
```

**Why this is useful:**
- **Multiple prompt variants** — easily switch between scoring approaches (e.g., `"scale"` vs `"category"`) via config
- **Model-specific formatting** — different models may need different system prompts or conversation structures
- **Safe formatting** — `copy.deepcopy` prevents accidental mutation of shared prompt templates

The prompt name is typically set in `settings.toml` and read at runtime:

```python
judge_prompt_name = settings.your_benchmark.judge_prompt_name
```

---

### HuggingFace Utilities

[`hf_utilities.py`](../src/mirroreval/hf_utilities.py) provides convenience wrappers for common HuggingFace operations:

| Function | Purpose |
|---|---|
| `call_hf_model(model_name, input_text)` | Quick single-call inference |
| `get_hf_pipeline(model_name, task)` | Get a reusable pipeline (better for batched calls) |
| `download_from_hf(repo_id)` | Pre-download a model/dataset for offline use |
| `load_hf_dataset(dataset_name)` | Load a dataset from HuggingFace Hub |
| `get_hf_model(model_name)` | Load an `AutoModel` |
| `get_hf_tokenizer(tokenizer_name)` | Load an `AutoTokenizer` |

Use `download_from_hf` in your entrypoint to ensure models are cached before SLURM jobs run without internet.

---

### SLURM Utilities

[`slurm_utilities.py`](../src/mirroreval/slurm_utilities.py) handles job submission on HPC clusters:

| Function | Purpose |
|---|---|
| `render_slurm_script(script_name)` | Renders a Jinja2 template with settings context |
| `submit_slurm_job(rendered_slurm_script)` | Submits the rendered script via `sbatch` |

SLURM parameters are configured in the `[slurm_job]` section of `settings.toml`. Your entrypoint should support both local and SLURM execution paths.

---

### Logger

A global logger is available for consistent output:

```python
from mirroreval.logger import logger

logger.info("Processing dataset...")
logger.warning("Missing score field in output")
logger.error("Unknown metric specified")
```

Use this instead of `print()` for all output.
