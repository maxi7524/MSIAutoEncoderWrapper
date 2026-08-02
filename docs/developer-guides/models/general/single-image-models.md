# Support single-image model contexts

A new family must define how one configured image dataset enters the graph and
how functionality is exposed in local context.

## Context scope and decisions

### Input contract

Choose raw/dense spectra, latent data, targets, and required `SpectrumSpace`.
Reuse `PixelDataset` when its sample contract is sufficient; add a registered
dataset only for genuinely different semantics.

### Local functionality

Define a family-specific runtime interface when inference requires operations
beyond direct `forward()`.

## Implementation instructions

### Connect dataset and graph

Validate input feature shape and target schemas during compile validation. Keep
reader and binner access through `active_context`; do not store wrapper globals
inside the model.

### Bind local runtime

Extend model-type detection and runtime-interface construction in
`ModelRuntimeProxy`. Local binding must store functionality in the selected
image ledger and survive loading another model globally.

### Test activation changes

Configure two images, bind the model to one, switch active image, and verify no
functionality leaks between ledger buckets.
