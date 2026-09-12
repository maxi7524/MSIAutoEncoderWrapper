# Add synthetic sampling strategies

These instructions extend the generated-spectrum strategies used by synthetic
pretraining. The module flow is described in
[Synthetic pretraining internals](../../../library-internals/models-training-and-execution/synthetic-pretraining.md).

## Add an annotation-coordinate strategy

Place another annotation-coordinate strategy in
`src/msi_autoencoder_wrapper/data/pretraining/strategies/with_annotations.py`
while the implementations remain small and share the same helper functions.
Create a separate module only when the strategy family has distinct inputs or
shared helpers.

Inherit `SyntheticSamplingStrategy` and register the implementation with a
stable configuration key:

```python
from msi_autoencoder_wrapper.data.pretraining import (
    SyntheticSampleDefinition,
    SyntheticSamplingStrategy,
    register_sampling_strategy,
)


@register_sampling_strategy("my_strategy")
class MySamplingStrategy(SyntheticSamplingStrategy):
    """Describe the generated component geometry and configuration semantics."""

    def __init__(self, component_count: int = 2) -> None:
        self.component_count = component_count

    def sample(self, rng, context, *, label_targets):
        components = ...
        return SyntheticSampleDefinition(
            components=tuple(components),
            label_targets=label_targets,
        )
```

`SyntheticSamplingManager.discover_strategies()` imports the strategy package.
The decorator validates the base class and registers the configuration key.
`get_available_strategies()` exposes the class docstring and constructor
parameters; document all configurable behavior in those two locations.

## Preserve strategy invariants

Use only the provided local `rng`; do not read or mutate global NumPy or Torch
random state. Return bin coordinates within `context.source.feature_count` and
use a target-column index only when it is aligned with
`context.source.class_names`. Preserve `label_targets` in the returned
definition. Do not render intensities, normalize spectra, or construct Torch
targets inside a strategy; those responsibilities belong to
`SyntheticSpectrumDataset`.

Constructor parameters are passed by `sampling_plan[*].parameters`. Validate
static configuration in the constructor. Validate constraints requiring the
current axis, eligible labels, or background bins during `sample()`.

## Add a new source family

Candidate-molecule strategies belong in
`data/pretraining/strategies/with_molecules.py`. Before registering one, add a
`SyntheticPeakSource` implementation that supplies binned coordinates and a
target vocabulary in the exact target-schema order. Update
`build_synthetic_partitions()` only when that source must become selectable from
a training configuration.

## Test the extension

Add focused tests under `tests/data/test_synthetic_pretraining.py`. Verify
determinism for fixed seed, output shape, finite nonnegative spectra, target
values and masks, parameter validation, and the intended behaviour when no
eligible labelled or background bins exist. Add an integration test under
`tests/training/` if the change modifies phase selection or partition handling.

Run:

```bash
UV_CACHE_DIR=/tmp/msi-autoencoder-uv-cache uv run pytest -q \
  tests/data/test_synthetic_pretraining.py tests/training/test_pretraining_phases.py
```

For phase configuration and loss selection, see
[Run synthetic spectral pretraining](../../../how-to/models-and-training/synthetic-pretraining.md).
