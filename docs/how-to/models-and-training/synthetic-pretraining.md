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

`sampling_plan` takes precedence over the legacy `modes` list. Its `count`
values must sum to `samples`. Validation retains the same strategy proportions;
the counts are deterministically scaled to `validation_samples`.

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
`annotated`, and `mixed`. `random`, `annotated`, and `mixed` accept
`min_peaks` and `max_peaks` inside their `parameters` mapping. An omitted
`max_peaks` uses phase-level `pretraining.max_peaks`.

The current training-phase builder derives labelled coordinates from the active
dataset's mapped annotation index and restricts eligible positive ions to the
real training split. Candidate-molecule sampling is not yet a training
configuration option.

For execution-wide phase, checkpoint, and loader options, see
[Train a model](training.md). For component flow, see
[Synthetic pretraining internals](../../library-internals/models-training-and-execution/synthetic-pretraining.md).
