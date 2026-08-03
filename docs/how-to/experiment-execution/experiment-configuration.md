# Configure an experiment

An experiment configuration expands one task definition into deterministic
repetitions and Cartesian parameter-grid variants.

## Purpose and available operations

### Configuration layers

The main YAML defines the experiment, task, runs, grid, reports, and execution
overrides. `execution.profile` may load a reusable local or Slurm profile whose
values are recursively merged with inline overrides.

### Reproducibility

Every repetition receives a deterministic seed derived from `base_seed`.
Different grid variants in the same repetition receive the same seed so paired
comparisons share controlled randomness.

## Detailed instructions

### Define all top-level fields

```yaml
schema_version: 1
experiment:
  name: example-campaign
task:
  entrypoint: package.module:run
  preflight_entrypoint: package.module:preflight
  parameters:
    training:
      batch_size: 32
runs:
  repetitions: 3
  base_seed: 42
grid:
  training.batch_size: [16, 32]
reproducibility:
  controls:
    - dataset_split
    - dataloader
    - sampler
    - augmentation
    - model_initialization
    - training
execution:
  profile: ../../assets/configs/execution/local.yaml
  max_parallel_runs: 1
reports: []
reporting:
  continue_on_error: true
```

`experiment.name` may contain letters, numbers, `.`, `_`, and `-`. Entrypoints
use `module:function`. Every grid key is a dotted path that must already exist in
`task.parameters`; every grid value is a non-empty list.

### Configure local execution

```yaml
execution:
  backend: local
  max_parallel_runs: 2
  work_directory: workspace/executions/example-campaign
```

`max_parallel_runs` is a positive integer. Without `work_directory`, output is
`workspace/executions/<experiment.name>` relative to the current directory.

### Configure Slurm execution

Use `assets/configs/execution/slurm.example.yaml` as the profile. Slurm requires
enabled staging and a staging root. Configure `partition`, `time`,
`cpus_per_task`, `memory`, `gpus_per_task`, and positive `array_parallelism`.
Staging input and result paths are resolved relative to the experiment
configuration directory and copied with checksum verification.

### Configure reports

`reports` is an ordered list of notebook paths or mappings containing `source`.
Reports run after local tasks or in the Slurm finalizer. `continue_on_error`
controls whether later reports run after one report fails.
