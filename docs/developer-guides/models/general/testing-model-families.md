# Test a model family

Family tests must cover registration through restored artifact execution.

## Required test coverage

### Contract coverage

Test category registration, incompatible implementations, duplicate conflicting
contracts, missing contracts, and discovery.

### Runtime coverage

Test graph assembly, forward output contract, single-image binding, cohort
identity, trained-state guards, and finite/domain-valid outputs.

## Implementation instructions

### Add a minimal family fixture

Use the smallest deterministic modules that exercise every category. Do not use
the production autoencoder as proof that another family contract works.

### Test artifact round-trip

Save configuration and weights, reconstruct in a fresh wrapper, and compare
state and output. Include strict-loading failure for incompatible weights.

### Enter registry-wide tests

The new family and each configurable component must participate in discovery,
configuration, and output-invariant parametrization.
