# MIRROR-Eval

MIRROR-Eval is an evaluation framework for MIRROR models.

## Installation

### Option 1: Install with pip

You can install the package directly from source:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

Or using requirements files:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### Option 2: Install with conda

Create a conda environment from the environment file:

```bash
conda env create -f environment.yml
conda activate mirror-eval
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
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
pip install dist/mirror_eval-0.1.0-py3-none-any.whl
```

## Usage

The primary entrypoint for MIRROR-Eval is the `evaluate` function:

```python
from mirror_eval import evaluate

# Run the evaluation pipeline
results = evaluate()
print(results)
```

## Development

### Project Structure

```
MIRROR-Eval/
├── src/
│   └── mirror_eval/
│       ├── __init__.py
│       └── evaluate.py
├── tests/
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── LICENSE
```

### Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=mirror_eval --cov-report=html
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

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.