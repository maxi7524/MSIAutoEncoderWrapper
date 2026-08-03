# Data representations

Pipeline stages use explicit immutable records so shape, device, axis, targets,
and normalization state move together.

## General abstraction

### Spectral stages

Raw spectra may have variable point counts or a shared native mass axis. Binning
creates dense `[B, F]` spectra. Encoding creates `[B, L]` latent values. Inverse
selection returns variable-length reconstructed spectra.

### Semantic descriptors

`SpectrumSpace` records mass axis, representation, normalization, and axis unit.
`TargetSchema` records target semantics once; `TargetBatch` carries values and
availability masks.

## Detailed implementation

### Raw batches

[`RawSpectrumBatch`](../../../src/msi_autoencoder_wrapper/data/batches.py) packs
points into one-dimensional tensors with offsets and sample indices.
`SharedAxisRawBatch` stores one mass axis and a `[B, P]` intensity matrix.

### Dense batches

`SpectrumBatch` validates one sample ID per row and feature count against
`SpectrumSpace`. Named views preserve the original spectra and target batch.

### Latent and inverse records

`LatentBatch` carries latent values, sample identity, targets, source space, and
normalization trace. `InverseSpectrumBatch` carries packed selected mass and
intensity values after reconstruction.
