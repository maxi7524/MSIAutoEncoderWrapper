# Instructions for developers

These documents describe how to modify, extend, validate, and maintain the
library. Internal execution relationships are documented under
[Library internals](../library-internals/index.md).

## Contents

- [Development environment](development-environment.md) — install the editable project and select repository checks.
- [Run and organize tests](testing.md) — choose test scope, reuse fixtures, and validate a change before review.
- [Configuration components](configuration-components.md) — implement one portable configuration contract and round-trip test.
- [Public exports](public-exports.md) — expose commonly constructed strategies without flattening internal modules.
- [Error handling](error-handling.md) — validate boundaries and use project exception categories.
- [Models](models/index.md) — add a model family or extend autoencoder implementations.
- [Objectives and metrics](objectives-and-metrics/index.md) — add losses and reusable metrics with compatibility requirements.
- [Data components](data-components/index.md) — add readers, binners, inverse binners, normalization, and datasets.
- [Dataset management](dataset-management/index.md) — add providers, provider filters, annotation normalization, and catalog behavior.
- [Analysis](analysis/index.md) — current rewrite requirements for adding analytical domains.

```{toctree}
:hidden:

development-environment
testing
configuration-components
public-exports
error-handling
models/index
objectives-and-metrics/index
data-components/index
dataset-management/index
analysis/index
adding-a-dataset-source
```
