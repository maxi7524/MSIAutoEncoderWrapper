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

## Normalization and reconstruction spaces

Normalization is configured per image through the context manager and executed
by ``active_context.normalization``. Steps are ordered, differentiable Torch
operations. Their inverse state is stored in ``NormalizationTrace`` and carried
by spectrum and latent batches.

```python
wrapper.context_manager.set_normalization(
    {
        "stage": "binned",
        "steps": {
            "tic": {"type": "tic", "epsilon": 1e-12},
        },
        "reconstruction": {
            "output_space": None,
            "denormalization_stage": "after_inverse_binning",
        },
    }
)
```

``stage`` is ``raw`` or ``binned`` and defaults to ``binned``. An omitted
``output_space`` derives behavior from that stage: raw normalization remains
normalized by default, while binned normalization reconstructs source
intensities. Call-level ``output_space`` overrides the active policy. Training
always consumes normalized tensors and does not apply the inference
reconstruction policy.

``set_normalization`` replaces the complete pipeline. ``update_normalization``,
``remove_normalization``, and ``clear_normalization`` provide explicit changes
without an implicit append operation. Saved image configuration includes step
order, parameters, stage, and reconstruction policy.

Dataset partitions are configured at the training root:

```python
training_config["dataset_split"] = {
    "train": 0.7,
    "validation": 0.15,
    "test": 0.15,
}
```

All three nonnegative proportions are required and must sum to one. Fitted
normalization state uses only the training partition. Validation selects the
best checkpoint; the test partition is evaluated once after the final phase.

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

Metrics may declare mathematical requirements including nonnegative values,
linear-intensity semantics, samplewise-scalar compatibility, and a required
output space. Normalization steps declare matching capabilities. The common
compatibility check rejects semantically invalid combinations before metric
execution. Masserstein, for example, accepts TIC, maximum, and L2 samplewise
scaling but rejects nonlinear intensity representations.
