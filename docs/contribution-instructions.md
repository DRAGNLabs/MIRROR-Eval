# Contributing to MIRROR-Eval

This markdown walks you, the contributor, through the process of preparing a new benchmark to be merged into the MIRROR-Eval pipeline.

It is expected that benchmark being contributed has already been developed in a separate codebase. Developing a benchmark within the MIRROR-Eval is likely not preferable.

A good example benchmark is [MTA](TODO connect to mta benchmark folder). See this example to understand the structure of a benchmark and it's components.

# Contribution Steps

## Create a branch

Create a new branch in the MIRROR-Eval repository. You will add your contribution to this branch and then create a pull request (PR) to merge it into the main branch.

TODO: add code exmaples on cloning, creating branch

## Setup MIRROR-Eval for development

You will want to have MIRROR-Eval setup for local development.

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

An easy way to test MIRROR-Eval is to run [demo.py](demo.py) # TODO: hook up file link

TODO: simple bash example for running demo.py after activating conda environment

Demo.py is very simple:

(TODO: encapsulate in code block)
import mirroreval

print(mirroreval.__version__)

config_path = "./settings.toml"

mirroreval.evaluate(config_path)

As can be seen we pass the configuration file path to the primary evaluate function in MIRROR-Eval. This configuration file is a crucial element. The configuration file specifies many global parameters for running the evaluation suite, including:
- The path to the model being evaluated
- Which benchmarks will be run
- Slurm parameters for running on a compute cluster
- Any specific parameters each benchmark consumes, such as dataset names, output directories, other model paths, etc.

See [settings.toml](settings.toml)(todo: hook up file link) to see how it is organized. We will modify this file in a later step to accommodate your new benchmark. For now, you should note that you can, and should, specify "model_checkpoint_path". For now, this parameter expects a HuggingFace model tag. A good default model to test with is Qwen/Qwen3-0.6B.

Before continuing, ensure that the pipeline can run by running demo.py:

## How the pipeline works

When you run demo.py with the default configuration file, MIRROR-Eval:
1. Collects which benchmarks are specified in the configuration file, and runs each one consecutively. In the case of the default configuration file, only one benchmark is ran: MTA.
2. The entrypoint.py file is run first. Each benchmark needs to contain an entrypoint script that 1) downloads any necessary models or datasets, and 2) begins running the benchmark by launching a Slurm job or just starting it locally. This step is crucial because high-performance computing systems usually do not have internet access so any models or data need to be preloaded.
3. Each benchmark does whatever it wishes to do to run the benchmark. In the case of MTA, a dataset is loaded, a conversation is simulated, this data is saved, an LLM-as-a-judge metric is run across the saved data, and analysis metrics such as mean score are computed.
4. Each benchmark returns a JSON data structure containing summary scores. The benchmark should also save these scores to a file, and is responsible for doing so.

## Contribution Requirements

To add a benchmark, you will need to complete the following steps:
1. Setup benchmark file structure
2. Add benchmark logic
3. Add a benchmark entry script
4. Update [settings.toml]() with benchmark-specific parameters
5. Create a benchmark markdown in [benchmarks]() that describes your benchmark.

We walk through each step in detail below:

### 1. Setup benchmark file structure

Create a directory within [benchmarks]() for your benchmark - give it the name of your benchmark. This directory should contain an empty __init__.py file.

### 2. Add benchmark logic

This directory will contain the bulk of your benchmark logic. Within this directory, you are free to organize your code in whichever manner you please. There are just a few core requirements:

- The benchmark needs to read "model_checkpoint_path" in the configuration file and load this model. This is the model that is being evaluated and it must be loaded by the benchmark itself.
- The benchmark needs to return a results JSON with the score. It must also save this JSON in the output directory specified in the configuration file.
- The benchmark needs to have an entrypoint script; see next step

The [MTA benchmark]() is a provided example benchmark. It is highly recommended that you review the code for this benchmark to understand the pipeline flow and how you might structure your benchmark to work efficiently.

### 3. Add a benchmark entry script

Each benchmark must have an entry script. This script has two purposes:
1. Download all necessary data and models before running the benchmark. This is required because compute nodes often do not have internet connection.
2. Launch the benchmark either locally or through Slurm, as specified in the configuration file. Slurm jobs, as run through high-performance computing systems, launch a seperate process.

It is recommended that you copy [mta_entrypoint.py]() and modify it to point towards your benchmark. You may or may not need to download models or datasets for your benchmark.

### 4. Update [settings.toml]() with benchmark-specific parameters

You benchmark is free to consume any parameters that the user specifies in the configuration files. Examples of parameters that you may wish to specify include:
- Datasets that are downloaded and used. MTA, for instance, specifies a dataset name.
- Models that are downloaded and used. MTA, for instance, specified a model name that is loaded for LLM-as-a-judge.
- Hyperparameters
- idk something else here

Update [settings.toml]() to match any parameters you reference in the config from your code. To access any parameters in the configuration file, you can import the settings object and directly access it. For example:

from mirroreval.config import settings
output_dir = Path(settings.mta.output_dir)

### 5. Create a benchmark markdown in [benchmarks]() that describes your benchmark.

Finally, provide documentation for your benchmark. Copy [example.md]() in [docs/benchmarks]() and fill out the sections as indicated. See [multiturn-accuracy.md]() as an example.

## Utilities

This section describes a few utilities and design patterns that are available to you to organize your benchmark code. These are utilized by the MTA metric, and so it is a good example to see how it is used.

### Datasets

TODO: describe dataset interfaces, how to implement the interface, and how to use the decorator and what it is useful for.

An abstract dataset interface is provided to act as a template for any dataset implementations.

A dataset should load the data, and provide a single line of data through the __getitem__ iterator function.
It is up to the metric classes to perform any necessary batching, etc.


Several elements are used to keep datasets flexible for multiple metrics/tasks:

- Item dataclass: The Item dataclass is used a return type, so that it can contain the 
  data as well as any important metadata such as split
- register_dataset decorator: This decorator is used to mark the class with a string that matches the dataset
  definition in the configuration file.

### Metrics

TODO: describe the metric interface, how to implement, and how to use the decorator.

### Prompts

TODO: describe the prompts pattern used in mta prompts and how this canbe useful.
