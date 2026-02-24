# Datasets

An abstract dataset interface is provided to act as a template for any dataset implementations.

A dataset should load the data, and provide a single line of data through the __getitem__ iterator function.
It is up to the metric classes to perform any necessary batching, etc.


Several elements are used to keep datasets flexible for multiple metrics/tasks:

- Item dataclass: The Item dataclass is used a return type, so that it can contain the 
  data as well as any important metadata such as split
- register_dataset decorator: This decorator is used to mark the class with a string that matches the dataset
  definition in the configuration file.
