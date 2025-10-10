# MIRROR-Eval

MIRROR-Eval is an evaluation framework for MIRROR models.
test
## Installation

Choose one of the following workflows based on your use case:

### For Users: Installing the Package

If you just want to use MIRROR-Eval in your projects:

**Using pip:**
```bash
# Clone the repository
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

# Install the package
pip install .
```

**Using conda:**
```bash
# Clone the repository
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

# Create and activate conda environment
conda env create -f environment.yml
conda activate mirror-eval
```

### For Developers: Setting Up the Development Environment

If you want to contribute to MIRROR-Eval or modify the code:

**Using pip:**
```bash
# Clone the repository
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Or install using requirements files
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

**Using conda:**
```bash
# Clone the repository
git clone https://github.com/DRAGNLabs/MIRROR-Eval.git
cd MIRROR-Eval

# Create and activate conda environment (installs package in editable mode)
conda env create -f environment.yml
conda activate mirror-eval

# Install development dependencies
pip install -r requirements-dev.txt
```

**Note:** The `-e` flag installs the package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

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
