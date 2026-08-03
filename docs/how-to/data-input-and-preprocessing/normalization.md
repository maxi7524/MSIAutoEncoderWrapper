# Configure normalization

Normalization applies ordered sample-wise transformations and records the state
required to restore source intensity scale.

## Purpose and available operations

### Supported normalization

The current built-in strategy is scalar normalization with `tic`, `max`, or
`l2` scaling. Pipelines operate at the `raw` or `binned` stage and declare where
inverse normalization occurs during reconstruction.

### Reconstruction policy

`output_space` is `normalized`, `source`, or `None` for the stage-derived
default. `denormalization_stage` is `after_decode` or
`after_inverse_binning`.

## Detailed instructions

### Set the complete pipeline

```python
wrapper.context_manager.set_normalization(
    {
        "stage": "binned",
        "steps": {
            "tic": {
                "type": "scalar",
                "kind": "tic",
                "epsilon": 1e-12,
            }
        },
        "reconstruction": {
            "output_space": "source",
            "denormalization_stage": "after_inverse_binning",
        },
    }
)
```

Step order follows mapping order. `epsilon` must be positive and protects empty
spectra from division by zero. Short forms named `tic`, `max`, or `l2` are
normalized to the scalar strategy.

### Update or remove steps

```python
wrapper.context_manager.update_normalization(
    {"steps": {"max": {"type": "scalar", "kind": "max"}}}
)
wrapper.context_manager.remove_normalization("max")
wrapper.context_manager.clear_normalization()
```

Updating rebuilds and validates the pipeline configuration. Clearing stores no
normalization pipeline for the selected image.

### Select reconstructed output space

```python
pipeline = wrapper.active_context.normalization
pipeline.set_output_space("source")
pipeline.set_denormalization("after_inverse_binning")
```

The pipeline rejects a denormalization location unsupported by any configured
step. Reconstruction requires the `NormalizationTrace` created for the same
pipeline and sample batch.
