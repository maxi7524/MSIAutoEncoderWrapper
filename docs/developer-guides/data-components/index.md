# Data components

These guides add implementations that transform MSI storage into model-ready or
reconstructed representations.

## Contents

- [Add a reader](adding-readers.md) — implement single and native batch reads while preserving the reader contract.
- [Add binners](adding-binners.md) — implement forward or inverse binning and validate axes and domains.
- [Add normalization](adding-normalization.md) — declare capabilities, state, inverse behavior, and configuration.
- [Add a model dataset](adding-model-datasets.md) — define samples, targets, partitions, contexts, and batching.

```{toctree}
:hidden:

adding-readers
adding-binners
adding-normalization
adding-model-datasets
```
