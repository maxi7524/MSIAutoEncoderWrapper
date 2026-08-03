# Reader-to-batch flow

The input path keeps storage I/O on CPU and performs vectorized preprocessing
after samples have been grouped into a batch.

## General abstraction

### Fast and fallback reads

Datasets request `GetSpectrumBatch()` when the reader supports native batching.
Shared-axis output avoids repeated mass arrays. Variable-axis output is packed
without padding.

### One preprocessing boundary

`BatchPreprocessor` owns device movement, forward binning, normalization, and
transfer to compute.

## Detailed implementation

### Read and collate

[`PixelDataset.get_raw_batch()`](../../../src/msi_autoencoder_wrapper/models/datasets/strategies/pixel_dataset.py)
converts native reader output into `SharedAxisRawBatch` or collates
`RawSpectrumSample` values through `RawSpectrumCollator`.

### Bin and normalize

[`BatchPreprocessor`](../../../src/msi_autoencoder_wrapper/data/preprocessing.py)
moves raw data to preprocessing device, applies raw-stage normalization when
configured, calls vectorized binner transformation, applies binned-stage
normalization, and attaches the trace.

### Deliver model input

The resulting `SpectrumBatch` moves to compute only when compute differs from
preprocessing. `model_input()` returns original spectra and named views in the
model contract expected by training.
