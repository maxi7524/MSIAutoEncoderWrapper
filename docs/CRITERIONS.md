# Training criteria

Criteria are organized by model family and by the part of the model output they
optimize. The registry has the following shape:

```text
model family
└── criterion type
    └── registered criterion name
```

The autoencoder family currently exposes:

- `reconstruction`: compares input spectra with `model_outputs["reconstruction"]`;
- `contrastive`: prepares paired inputs and evaluates `model_outputs["projection"]`;
- `head`: reserved for objectives consuming a named `head_<name>` output and its
  dataset target.

## Training configuration

Use explicit categories in every phase:

```python
training_config = {
    "phases": [
        {
            "phase_name": "joint",
            "epochs": 20,
            "batch_size": 64,
            "criterions": {
                "reconstruction": {
                    "mse": {
                        "target": "MSELoss",
                        "weight": 1.0,
                        "params": {},
                    },
                },
                "contrastive": {
                    "info_nce": {
                        "target": "InfoNCELoss",
                        "weight": 0.05,
                        "params": {"temperature": 0.07},
                    },
                },
            },
        }
    ]
}
```

A weighted `CompositeLoss` evaluates every configured component and reports both
individual values and `total_loss`. A flat legacy mapping remains accepted when
every registered target belongs unambiguously to one category.

## Lifecycle hooks

`on_phase_start(model, dataset, transient_cache)` runs once before a phase. It is
intended for reusable computations. `InfoNCELoss`, for example, samples spectra,
detects peaks, measures their envelopes, and stores a bounded peak bank in the
transient cache.

`on_batch_start(batch_data, transient_cache)` runs before the model forward pass.
The contrastive criterion uses it to inject sampled foreign peak envelopes and to
stack original and augmented spectra into a `2N` batch. Reconstruction criteria
then compare the corresponding `2N` reconstructions against the same prepared
batch.

## Masserstein reconstruction loss

`MassersteinLoss` is a differentiable adaptation of the spectral regression
method described in:

> Ciach et al., *Masserstein: linear regression of mass spectra by optimal
> transport*, Rapid Communications in Mass Spectrometry, DOI `10.1002/rcm.8956`.

The important operations are:

1. Negative input and reconstruction values are clamped to zero because transported
   ion current must be non-negative.
2. The two spectra are divided by their joint maximum total ion current. Their
   relative mass difference is retained; they are not normalized independently.
3. Each spectrum is augmented with the other spectrum's total mass at an auxiliary
   point `ω`. Both augmented measures therefore have equal mass.
4. Real-to-real transport costs the physical m/z distance. Moving signal between
   a real bin and `ω` costs the denoising penalty `κ`; `ω`-to-`ω` transport is free.
5. Signal farther apart than `2κ` is cheaper to destroy and recreate through `ω`
   than to transport directly. This is a soft optimal-transport decision, not a
   hard peak-matching window.
6. A log-domain entropy-regularized Sinkhorn solver obtains the transport plan.
7. Sinkhorn self-costs can be subtracted to remove entropy bias. Gradients are
   propagated through the converged dual potentials, avoiding storage of every
   solver iteration in the autograd graph.

Example:

```python
"reconstruction": {
    "masserstein": {
        "target": "MassersteinLoss",
        "weight": 1.0,
        "params": {
            "denoising_penalty": 0.4,
            "entropy_regularization": 0.02,
            "sinkhorn_iterations": 50,
        },
    }
}
```

`denoising_penalty` uses the same unit as the spectral axis. During training the
criterion reads `binner.GetXAxis()` when available. Without a binner axis it uses
uniform positions separated by `axis_step`.

Lower entropy regularization and more Sinkhorn iterations improve approximation
accuracy but increase runtime. The ground-cost matrices are quadratic in the
number of spectral bins. Run `models_manager.estimate_training_resources(...)`
before using this loss on a high-resolution grid and monitor the first epoch.
