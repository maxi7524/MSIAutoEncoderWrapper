# Add an autoencoder graph

An autoencoder graph coordinates registered encoder, decoder, optional
projector, and named heads while preserving the family output dictionary.

## Autoencoder scope and contracts

### Master responsibilities

The master routes model input through encoder and decoder, retains latent
values, executes heads, and implements backbone freezing. Component internals
remain in their own classes.

### Output contract

Training criteria depend on stable reconstruction, latent, variational, and
`head_<id>` keys. A new graph must document any additional outputs.

## Implementation instructions

### Implement and register

Inherit `MSIBaseAutoencoderArchitecture` and register the class as the
`autoencoder` master only if it replaces the selected family graph. If several
master implementations must coexist, the architecture manager requires a
separate implementation-selection dimension rather than overwriting one key.

### Accept resolved components

Consume `resolved_components` categories and `head_specs`. Validate required
encoder/decoder presence and head bindings during construction.

### Preserve freezing and configuration

Implement `freeze_backbone()` consistently and export component descriptors and
model parameters required by `ModelLoader`.
