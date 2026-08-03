# Models and training

These guides configure model input datasets, model architectures, training
phases, and persistent artifacts.

## Contents

- [Configure model datasets](datasets.md) — choose single-image or cohort spectra, targets, normalization, and splitting.
- [Configure a model](model-configuration.md) — select a model family, components, heads, or preset.
- [Configure autoencoders](autoencoder-models.md) — assemble encoder, decoder, projector, and classification heads.
- [Configure cohort models](cohort-models.md) — train or use models over multi-image contexts.
- [Train a model](training.md) — define phases, losses, optimizers, devices, loaders, and checkpoints.
- [Persist and restore models](model-persistence.md) — save, load, bind, and export complete model artifacts.

```{toctree}
:hidden:

datasets
model-configuration
autoencoder-models
cohort-models
training
model-persistence
```
