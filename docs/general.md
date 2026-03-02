# General

TODO: gotta change this, this is an outdated model of how this should run

MIRROR-Eval is a benchmark pipeline for running a single model across multiple tasks.

The pipeline follow a specific composition: it iterates through a list
of benchmarks, each of which are composed of a dataset and multiple metrics.
Thus, the scores returned are composed of multiple benchmarks, each of which
have multiple metric scores on a single task.

- Benchmarks
  - Dataset
  - Metrics

A basic workflow for a benchmark may be:

1. Load model, load data
2. Run model on data, if necessary
3. Run each metric on model output
4. Save all results to a given results folder

All results for a benchmark should be stored in a single file,
where each line is a single data point. This file is important for further manual analysis.

All scores, calculated for the benchmark, should be stored in another file.
For example, accuracy, confidence intervals, etc. The structure of this file can vary.
