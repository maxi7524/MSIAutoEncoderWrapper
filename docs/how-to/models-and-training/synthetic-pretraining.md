# Run synthetic spectral pretraining

Synthetic pretraining adds one or more generated-spectrum phases before real-data
adaptation. It uses the active model axis and target schema, then preserves model
weights when the next real-data phase begins.

## Configure a synthetic phase

Place `pretraining` on a regular training phase. `samples` is the number of
synthetic training examples per epoch. `validation_samples` configures a separate
deterministic synthetic validation population.

```python
training = {
    "seed": 42,
    "checkpoint": {"enabled": True, "restore_best": True},
    "phases": [
        {
            "phase_name": "synthetic_pretraining",
            "epochs": 15,
            "batch_size": 64,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-3, "weight_decay": 1e-4},
            },
            "pretraining": {
                "samples": 4096,
                "validation_samples": 512,
                "seed": 42,
                "max_peaks": 16,
                "normalization": "tic",
                "sampling_plan": [
                    {
                        "strategy": "single_random",
                        "count": 1024,
                        "label_targets": False,
                    },
                    {
                        "strategy": "single_annotated",
                        "count": 1024,
                    },
                    {
                        "strategy": "annotated",
                        "count": 1024,
                        "parameters": {"min_peaks": 2, "max_peaks": 6},
                    },
                    {
                        "strategy": "mixed",
                        "count": 1024,
                        "parameters": {"max_peaks": 12},
                    },
                ],
            },
            "criterions": {
                "reconstruction": {
                    "mse": {"target": "MSELoss", "weight": 1.0},
                },
                "heads": {
                    "molecule_primary": {
                        "bce": {"target": "MultiLabelBCELoss", "weight": 0.2},
                    },
                },
            },
        },
        {
            "phase_name": "real_adaptation",
            "epochs": 15,
            "batch_size": 64,
            "criterions": {
                "reconstruction": {
                    "mse": {"target": "MSELoss", "weight": 1.0},
                },
            },
        },
    ],
}
```

`sampling_plan` takes precedence over the legacy `modes` list. Its generated
counts must sum to `samples`. For `candidate_permuted`, the generated count is
`count * parameters.permutations`. Validation retains the same strategy
proportions; the counts are deterministically scaled to `validation_samples`.

Every phase uses a registered `representation` to convert declared components
into spectra. The default `triangular_peak` preserves historic behaviour. A
sampling-plan entry may override it, allowing composition sampling and spectrum
rendering to vary independently:

```yaml
pretraining:
  representation:
    strategy: triangular_peak
    parameters:
      peak_radius: 1
  sampling_plan:
    - strategy: annotation_uniform_mixture
      count: 4096
      representation:
        strategy: triangular_peak
        parameters:
          peak_radius: 2
```

`isospec_envelope` is an alternative renderer for components that declare a
`formula|adduct` molecular identity (or equivalent source metadata). It uses
IsoSpec to enumerate a natural isotope envelope, applies the final ionic charge
correction, maps all fine-structure lines with the active binner, and sums the
result before normalisation. Relative abundances of molecular components are
drawn from a Dirichlet occupancy and log-normal response model.

Use the renderer only for strategies that generate declared molecular
components. Keep unlabelled/background coverage on `triangular_peak` (or a
future background renderer), because an unlabelled bin has no formula from
which an isotope envelope can be derived.

```yaml
pretraining:
  representation:
    strategy: isospec_envelope
    parameters:
      probability_coverage: 0.999
      max_isotopologues: 4096
      occupancy_alpha: 1.0
      intensity_log_sigma: 0.5
  sampling_plan:
    - strategy: annotation_axis_coverage
      count: 1024
      label_targets: true
      representation:
        strategy: triangular_peak
    - strategy: annotation_uniform_mixture
      count: 3072
      parameters: {min_peaks: 15, max_peaks: 30}
```

## Select objectives and labels

Reconstruction criteria consume every generated spectrum. A head criterion
consumes a synthetic target only when its target mask is available.

`label_targets: false` retains the generated spectrum but masks every synthetic
target. Use it for reconstruction-only examples, or omit head criteria from the
phase entirely. With the default `label_targets: true`, labelled components are
positive molecular targets and generated background components are known
negatives for the current molecular vocabulary.

`element_counts` is available only for a single labelled component without a
background component. Mixed examples therefore do not contribute to
`ElementCountLoss`.

## Use available strategies

The current strategies are `single_random`, `single_annotated`, `random`,
`annotated`, `mixed`, `candidate_single`, `candidate_permuted`, and
`candidate_convolved`. `random`, `annotated`, and `mixed` accept `min_peaks`
and `max_peaks`. Candidate strategies accept molecule-count, class-selection,
occupancy, and intensity parameters inside their `parameters` mapping.

Pixel-aware train-annotation strategies are `annotation_axis_coverage`,
`annotation_uniform_mixture`, `annotation_overlap_mixture`, and
`annotation_rare_class_mixture`. They retain the source pixel and complete
local annotation set for every selected bin. `annotation_axis_coverage` uses a
count divisible by the binned-axis width, giving identical bin coverage across
the strategy population. The remaining strategies generate mixtures of 15--30
components by default and use train-only annotation records.

## Generate spectra from the candidate catalog

Set `peak_source` to `candidate_catalog` when the active dataset has a loaded
`candidates.sqlite` catalog:

```python
"pretraining": {
    "samples": 4096,
    "validation_samples": 512,
    "peak_source": "candidate_catalog",
    "candidate_filters": {
        "polarity": "Positive",
        "mz_min": 100,
        "mz_max": 1500,
    },
    "candidate_classes": ["Fatty Acyls", "Organic compounds"],
    "candidate_label_targets": False,
    "sampling_plan": [
        {
            "strategy": "candidate_single",
            "count": 1024,
        },
        {
            "strategy": "candidate_permuted",
            "count": 512,
            "parameters": {
                "permutations": 4,
                "min_molecules": 2,
                "max_molecules": 6,
                "occupancy_alpha": 0.8,
                "intensity_log_sigma": 0.5,
            },
        },
        {
            "strategy": "candidate_convolved",
            "count": 1024,
            "parameters": {
                "min_molecules": 2,
                "max_molecules": 8,
                "class_balanced": True,
            },
        },
    ],
}
```

`candidate_permuted` draws an occupancy vector from a Dirichlet distribution
and independent component intensities from a log-normal distribution. The
weighted components are normalized by the phase normalization setting. The
generation parameters and component provenance are available in
`SpectrumBatch.metadata["synthetic_generation"]`.

`candidate_convolved` is the weighted sum of candidate peak profiles. It does
not impose biological co-occurrence constraints. When
`candidate_label_targets` is `True`, candidate labels are limited to the
current molecule target vocabulary. Set it to `False` for reconstruction-only
candidate generation with masked targets.

The default `annotation` peak source derives labelled coordinates from the
active dataset's mapped annotation index and restricts eligible positive ions
to the real training split. The `candidate_catalog` source uses the loaded
external candidate catalog and the active binner instead.

For execution-wide phase, checkpoint, and loader options, see
[Train a model](training.md). For component flow, see
[Synthetic pretraining internals](../../library-internals/models-training-and-execution/synthetic-pretraining.md).
