# MIRROR-Eval

MIRROR-Eval is an evaluation framework for MIRROR models.

## Installation

```bash
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

conda env create -f environment.yml
conda activate mirror-eval

pip install .
```

## Usage

Pass a configuration file to the `evaluate` function:

```python
from mirroreval import evaluate

evaluate("settings.toml")
```

Or run [demo.py](demo.py) directly:

```bash
python demo.py
```

See [settings.toml](settings.toml) for available configuration options.

## Contributing

See the [contribution guide](docs/contribution-instructions.md) for instructions on adding benchmarks, setting up for development, and submitting a PR.

## Tips

Set the HuggingFace cache directory if the default location isn't suitable:

```bash
export HF_HOME="/path/to/cache/dir"
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
