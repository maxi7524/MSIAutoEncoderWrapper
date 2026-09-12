# Models, training, and execution

This section describes architecture assembly, loaded and local runtime models,
the training lifecycle, and campaign execution.

## Contents

- [Model architecture system](model-architecture-system.md) — family-owned contracts, component registries, presets, and graph assembly.
- [Model runtime](model-runtime.md) — attachment, trained state, local binding, and artifact loading.
- [Training lifecycle](training-lifecycle.md) — partitions, preprocessing, phases, losses, checkpoints, and histories.
- [Synthetic pretraining internals](synthetic-pretraining.md) — generated-spectrum sources, strategies, targets, and phase transitions.
- [Experiment execution](experiment-execution.md) — configuration merge, deterministic planning, local/Slurm backends, staging, and reporting.

```{toctree}
:hidden:

model-architecture-system
model-runtime
training-lifecycle
synthetic-pretraining
experiment-execution
```
