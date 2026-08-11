# Binning and normalization

Binning defines a shared feature axis; normalization defines intensity scale and
the reversible state carried through model execution.

## General abstraction

### Transformation order

Raw-stage normalization runs before binning. Binned-stage normalization runs
after dense aggregation. The chosen stage is stored in the trace and affects
valid reconstruction order.

### Capability intersection

Every normalization step declares invertibility, non-negativity, linearity,
sample-wise scalar behavior, and supported inverse locations. A pipeline exposes
the conservative intersection.

## Detailed implementation

### Vectorized binning

[`LinearBinning.transform()`](../../../src/msi_autoencoder_wrapper/binners/binners_strategies/linear_binner.py)
calculates integer bin indices and uses Torch scatter aggregation. Shared-axis
batches map the axis once and scatter all rows.

### Normalization trace

[`NormalizationPipeline`](../../../src/msi_autoencoder_wrapper/normalization/pipeline.py)
applies ordered steps and stores one state mapping per step. Inverse executes
steps in reverse order and requires matching trace length.

### Compatibility

Metric and reconstruction paths must request representations supported by the
pipeline capabilities. Unsupported post-inverse-binning denormalization fails
during policy configuration.
