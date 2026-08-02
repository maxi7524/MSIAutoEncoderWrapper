# MSI data pipeline

This document defines the internal data representations and device boundaries
shared by preprocessing, training, inference, and analysis.

## Representation stages

Raw spectra have variable point counts and use ``RawSpectrumBatch``. Its m/z
positions and intensities are packed into one-dimensional tensors. ``offsets``
and ``sample_indices`` preserve sample boundaries without padding.

Forward binning produces ``SpectrumBatch``. Its ``spectra`` tensor has shape
``[B, F]`` and its ``SpectrumSpace.mass_axis`` has shape ``[F]``. The axis is a
shared descriptor; it is not expanded or copied for every spectrum.

Latent tensors have shape ``[B, L]`` and are not defined on an m/z axis. A
latent representation may retain the source or reconstruction
``SpectrumSpace`` as metadata, but latent dimensions must not be interpreted as
m/z coordinates.

Inverse binning returns variable-length spectra. These results use the same
packed representation as raw spectra instead of padding every result to a
common length.

## Device boundaries

Readers perform storage I/O on CPU. DataLoader workers pack raw points in CPU
memory. The main process moves ``RawSpectrumBatch`` to
``preprocessing_device`` and executes the Torch binner there. The resulting
dense batch is moved to ``compute_device`` only when the devices differ.

Training configuration accepts both settings globally or per phase:

```python
training_config = {
    "preprocessing_device": "cuda",
    "compute_device": "cuda",
    "phases": [
        {
            "preprocessing_num_workers": {"cpu": 4, "cuda": 2},
            "dataloader": {},
        }
    ],
}
```

When omitted, both resolve to the wrapper device. CPU and CUDA preprocessing
call the same Torch implementation; this permits numerical comparison and
device-specific benchmarking without separate algorithms.
``preprocessing_num_workers`` selects a default by preprocessing device. An
explicit ``dataloader.num_workers`` value takes precedence. CUDA preprocessing
still permits reader workers because workers only perform storage I/O and raw
packing; CUDA allocation remains in the main process.

## Targets and views

``TargetBatch`` carries values and availability masks with the same leading
sample dimension as ``SpectrumBatch``. ``TargetSchema`` stores class names and
target semantics once and remains on CPU.

Augmentations add named tensors to ``SpectrumBatch.views``. They do not replace
the original spectra or remove targets and masks. The model runner may combine
views for one forward operation while preserving the logical sample count
``B``.

## Metrics and criteria

The top-level ``metrics`` package owns numerical implementations grouped by
object space: spectra, classification decisions, classes, and embeddings.
Criteria select tensors from model and batch contracts, invoke a fixed metric,
and reduce its output for optimization. Criteria must not modify batches.

Analysis invokes the same metrics under inference mode. Per-sample metrics are
transferred to CPU only when retained or visualized. Dataset and class metrics
accumulate sufficient statistics across batches instead of averaging
independently calculated batch values.
