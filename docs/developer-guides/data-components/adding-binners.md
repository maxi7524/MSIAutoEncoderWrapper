# Add forward or inverse binning

Forward binners define dense model features; inverse binners select a sparse
representation from those features.

## Binner scope and invariants

### Forward contract

Inherit `MSIBaseBinner`, expose a stable axis and bounds, implement scalar-call
and vectorized batch transformation, and preserve batch identity and targets.

### Inverse contract

Inherit `MSIBaseInverseBinner`, use the matching forward axis, and return finite
mass/intensity arrays. Selection must be deterministic for fixed input and
configuration.

## Implementation instructions

### Register the implementation

Use `register_binner()` or `register_inverse_binner()` in the responsible
strategy package. Constructor state must use the shared configuration system.

### Validate domains and budgets

Reject invalid axes, non-positive bin widths, negative/non-finite inputs when
the strategy assumes intensity mass, and inconsistent minimum/maximum budgets.
Expose selection diagnostics when they affect interpretation.

### Add tests

Compare scalar and vectorized forward results, CPU/CUDA results where available,
empty spectra, boundary masses, non-finite input, inverse selection counts,
non-negativity, and round-trip configuration.
