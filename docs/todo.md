notes:



A major challenge ends up being efficiency: this pipeline is basically a bunch of nested loops. Picking the right loop order is critical:

old process:
for each benchmark in config benchmarks:
    download all necessary data or models
    launch evaluation locally or through slurm (renders script).
        iterate through datasets and metrics
        pass data through metrics, save score?
        Compute scores from saved metrics. Like accuracy, other stuff.
        

new process:
for each benchmark in config benchmarks:
    download all necessary data/models for evlauation stuff
    launch evaluation locally or through slurm -> pass just the model being tested.
        SOMETHING. Each benchmark is a black box, it has the liberty to do whatever it wants, such as:
        - iterate through different metrics, comparative data, run multiple judges, etc.
        - compile scores, run additional statistical tests, etc.
    OUT: each benchmark should return a data structure representing scores. It doesn't have to be a single score, could have confidence intervals, multiple scores, etc.
    

TODO:

adapt the creativity benchmark into generalized accuracy, using a single dataset with basic LLM-as-a-judge and cos sim
create contribution instrucitons from this. 
A contribution needs ot have input/output documentation, general documentation, details on requirements,
A contribution should add a doc outlining these specific things/requirements.


develop a standardized output json class to inherit from.
set this as the output of the primary benchmark class.


Figure out how to pass a model to the benchmark:
let's start by use case: youre local or on a login node and you call evaluate which either triggers jobs or just does the benchmarks in order. So the settings should have a checkpoint path. Loading the model is up to the individual benchmarks.
should each benchmark load and run the model from checkpoint seperately?
Need three ways to run this thing:
1. Just plan run on a local gpu/hardware 
2. Through slurm, meaning a slurm process is launched. to run a model on a benchmark
3. on an existing running slurm process..this would be the same as 1?
