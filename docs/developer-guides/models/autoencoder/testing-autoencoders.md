# Test autoencoder components

Autoencoder tests enforce component contracts and MSI-specific reconstruction
invariants.

## Required test coverage

### Component behavior

Test input/output shape, finite values, gradient flow, configuration round-trip,
and registry resolution for each encoder, decoder, projector, and head.

### MSI invariants

Decoder final output must be non-negative. Reconstruction losses must reject
negative intensity targets or outputs when their metric requires non-negative
spectra. Variational paths must retain expected parameters.

## Implementation instructions

### Parametrize registered components

Discover architectures and iterate each autoencoder category. Supply minimal
valid parameters per implementation and apply shared output assertions.

### Test full graph paths

Cover plain reconstruction, variational encoding, projector output, multiple
heads, training checkpoint, runtime encode/decode, and saved artifact round-trip.
