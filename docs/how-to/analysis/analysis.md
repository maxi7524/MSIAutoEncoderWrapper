# Analysis status

The current autoencoder analysis API is a working, provisional implementation.
It must not be treated as a stable public contract until the analysis rewrite is
complete.

## Purpose and available operations

### Current scope

`AutoencoderAnalysis` and `AutoencoderMultiAnalysis` prepare retained model
outputs and expose reconstruction, latent, head, binning, metric, and plotting
operations.

### Required responsibility boundary

Analysis coordinates analytical logic. Metric calculation belongs in
`metrics`; reusable rendering belongs in `visualization`; analysis should
aggregate those operations for a defined question.

## Detailed instructions

### Provisional preparation flow

The current API requires `prepare()` before operations that consume retained
inputs, reconstructions, latents, outputs, targets, masks, or coordinates.
`estimate_prepare_size()` estimates retained memory before materialization.

### Stability warning

Method names, retained-result layout, and analysis object composition may change
during the rewrite. New user workflows should not depend on undocumented
analysis internals. This page will be replaced by task-specific guides after the
module responsibilities and result contracts are finalized.
