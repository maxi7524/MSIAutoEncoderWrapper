# Add an encoder or decoder

Encoders map dense MSI features to latent representations; decoders map latent
representations back to the configured feature axis.

## Component scope and invariants

### Encoder contract

Inherit `MSIBaseEncoder`, accept batched `[B, F]` input, and return the latent
form expected by the master graph. Variational encoders must expose distribution
parameters through the established output contract.

### Decoder contract

Inherit `MSIBaseDecoder`, return `[B, F]`, and apply a configured output
activation. MSI reconstructed intensity must not be negative.

## Implementation instructions

### Register the component

```python
@ArchitecturesManager.register_component("autoencoder", "encoder", "MyEncoder")
class MyEncoder(MSIBaseEncoder):
    ...
```

Use category `decoder` and `MSIBaseDecoder` for decoders. Store all constructor
state through the configurable-component contract.

### Validate dimensions and domain

Reject inconsistent layer dimensions during construction. Test batch sizes 1
and greater than 1, finite output, expected latent/feature shape, gradients, and
non-negative final decoder output for negative pre-activation values.
