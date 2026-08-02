# Add normalization

A normalization step transforms dense or packed values and returns state needed
for inverse transformation.

## Normalization scope and capabilities

### Required interface

Inherit `BaseNormalization`, implement `transform()`, `inverse()`, and
`get_config()`, and declare `NormalizationCapabilities`.

### Capability meaning

Capabilities state invertibility, preservation of non-negativity and linear
intensity, sample-wise scalar behavior, and allowed inverse locations.

## Implementation instructions

### Implement dense and packed paths

Support `[B, F]` dense input. Support packed `[N]` input with sample indices and
batch size when the step can run at raw stage. Return state tensors with one
sample dimension where appropriate.

### Integrate configuration

Add strategy parsing to `NormalizationPipeline.from_config()` or replace its
current explicit mapping with a normalization registry when multiple families
justify discovery. Preserve ordered step names and parameters.

### Test inverse and compatibility

Check dense/packed equivalence, empty spectra, epsilon behavior, device movement,
transform/inverse recovery, capability intersection, unsupported reconstruction
policy, and configuration round-trip.
