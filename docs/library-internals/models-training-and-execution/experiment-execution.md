# Experiment execution

Experiment execution expands one campaign configuration into deterministic
tasks and routes the same task files to local or Slurm backends.

## General abstraction

### Planning before execution

Configuration validation and preflight occur before task materialization.
Planning is backend-independent; backends consume identical task descriptors.

### Persistent and staged state

The plan directory is persistent. Slurm may copy plans and declared inputs to a
node-local staging root, then checksum-restore declared results.

## Detailed implementation

### Load and merge configuration

[`runtime/configuration/loading.py`](../../../src/msi_autoencoder_wrapper/runtime/configuration/loading.py)
loads schema-v1 YAML and recursively merges an optional execution profile.
Internal `_config_path` and `_config_directory` fields retain resolution roots.

### Build deterministic tasks

[`build_plan()`](../../../src/msi_autoencoder_wrapper/runtime/planning/plan.py)
calculates the Cartesian grid and repetitions. A SHA-256-derived repetition seed
is shared across grid variants in the same repetition. Dotted grid paths must
already exist in task parameters.

### Execute backends

Local execution bounds concurrent tasks. Slurm writes an array script and a
dependent finalizer. Status manifests transition through running, completed, or
failed and retain task/result context.

### Stage and report

[`staging.py`](../../../src/msi_autoencoder_wrapper/runtime/staging.py)
copies declared paths with checksum verification and guards cleanup by execution
identity. Reporting executes ordered notebooks and writes a report manifest.
