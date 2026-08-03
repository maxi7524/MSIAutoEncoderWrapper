# Data pipeline

This section describes representation and device transitions from reader output
to model input and reconstructed spectra.

## Contents

- [Data representations](representations.md) — raw, dense, latent, inverse, target, and spatial contracts.
- [Reader-to-batch flow](reader-to-batch-flow.md) — native batch reads, packing, binning, and normalization.
- [Devices and workers](devices-and-workers.md) — CPU I/O, preprocessing device, compute device, and transfer boundaries.
- [Binning and normalization](binning-and-normalization.md) — ordered transformation state and capability contracts.
- [Targets and annotations](targets-and-annotations.md) — target schemas, masks, class mappings, and spectrum links.
- [Reconstruction flow](reconstruction-flow.md) — decode, output activation, denormalization, and inverse binning.

```{toctree}
:hidden:

representations
reader-to-batch-flow
devices-and-workers
binning-and-normalization
targets-and-annotations
reconstruction-flow
```
