# Training lifecycle

Training converts one configured model and dataset into ordered optimization
phases, partition metrics, checkpoints, and persistent history.

## General abstraction

### Manager and engine

`TrainingManager` retains portable training configuration and shared transient
loss cache. `MSIPyTorchTrainer` owns partitions, DataLoaders, devices,
optimization loops, validation, early stopping, and checkpoints.

### Criterion graph

`CriterionsManager` builds a composite loss from model-family registrations.
Criteria select tensors from model outputs and `SpectrumBatch`; numerical
metric definitions remain outside the trainer.

## Detailed implementation

### Prepare partitions and normalization

[`MSIPyTorchTrainer.fit()`](../../../src/msi_autoencoder_wrapper/training/engine/base_trainer.py)
validates active state, creates dataset partitions, and fits context
normalization using the training partition only.

### Prepare each phase

For every phase the engine resolves compute and preprocessing devices, applies
freeze paths, builds composite loss, creates a Torch optimizer, runs criterion
phase hooks, and builds training and optional validation loaders.

### Process batches

Raw batches are moved once to the preprocessing device, binned and normalized
by `BatchPreprocessor`, then moved to compute. Criterion batch hooks may add
views. The engine checks inputs, model outputs, loss values, and gradients for
finite values before optimizer updates.

### Select and persist best state

Validation loss updates best-state and patience counters. Enabled checkpoints
save schema-v2 configuration, weights, and history. `restore_best` reloads the
selected state after phases complete. History entries identify best epochs and
retain phase metrics.
