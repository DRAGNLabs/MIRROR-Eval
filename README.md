# MIRROR-Eval

MIRROR-Eval is an evaluation framework for MIRROR models.

## Installation for development

**Using conda:**

```bash
# Clone the repository
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

# Create and activate conda environment
conda env create -f environment.yml
conda activate mirror-eval

pip install .

# If you're doing development:
pip install -e ".[dev]"

```

## Building the Package

### Build from source

To build the package as a distributable wheel:

```bash
# Install build tools
pip install build

# Build the package
python -m build
```

This will create distribution files in the `dist/` directory:
- A wheel file (`.whl`) for binary distribution
- A source distribution (`.tar.gz`)

### Installing the built package

```bash
pip install dist/mirroreval-0.1.0-py3-none-any.whl
```

## Usage

The primary entrypoint for MIRROR-Eval is the `evaluate` function:

```python
from mirroreval import evaluate

# Run the evaluation pipeline
evaluate("settings.toml")
```

## Demo

To test it out, try running [demo.py](demo.py)

## Development

### Project Structure

```
MIRROR-Eval/
├── src/
│   └── mirroreval/
│       ├── __init__.py
│       ├── evaluate.py
│       ├── config.py
│       ├── logger.py
│       ├── hf_utilities.py
│       ├── slurm_utilities.py
│       ├── slurm_templates/
│       └── benchmarks/
│           └── creativity/
│               ├── creativity_entrypoint.py
│               ├── creativity_benchmark.py
│               ├── creativity_metrics.py
│               ├── creativity_datasets.py
│               ├── creativity_analysis.py
│               └── prompts.py
├── tests/
├── notebooks/
├── pyproject.toml
├── environment.yml
├── settings.toml
├── README.md
└── LICENSE
```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=mirroreval --cov-report=html
```

### Code Formatting

Format code with Black:

```bash
black src/
```

Check code style with flake8:

```bash
flake8 src/
```

Type checking with mypy:

```bash
mypy src/
```

## Other tips

Set the HuggingFace cache on your machine:

```bash
export HF_HOME="/path/to/cache/dir"
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
