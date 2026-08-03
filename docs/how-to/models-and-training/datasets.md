# Configure model datasets

Model datasets turn image or cohort contexts into samples, targets, and stable
train/validation/test partitions.

## Purpose and available operations

### Dataset families

`PixelDataset` reads one active image. `CohortPixelDataset` concatenates original
spectra from cohort members. `CohortLatentDataset` concatenates previously
materialized latent spectra.

### Dataset responsibilities

Datasets select image or latent input, attach annotation-derived targets, expose
stable sample IDs, and create partitions from a split configuration.

## Detailed instructions

### Configure a single-image dataset

```python
wrapper.models_manager.set_dataset(
    "PixelDataset",
    source="image",
    normalization="tic",
    normalization_epsilon=1e-12,
    target_specs={},
    split={"strategy": "random", "seed": 0},
)
```

- `source` is `image` or `latent`;
- `normalization` is `none`, `tic`, `max`, or `l2`; image input defaults to
  `tic`, latent input defaults to `none`;
- `normalization_epsilon` must be positive;
- `target_specs` maps target names to single-label or multi-label definitions;
- `split` configures partition strategy and fractions.

Dataset-local normalization is retained for the current dataset API. The
context normalization pipeline controls the batch preprocessing and
reconstruction path; do not configure two independent scalings accidentally.

### Configure annotation targets

```python
target_specs = {
    "molecule": {
        "type": "multi_label",
        "class_mapping": {
            "C6H12O6|+H": 0,
        },
    },
    "condition": {
        "type": "single_label",
        "class_mapping": {"healthy": 0, "disease": 1},
    },
}
```

`molecule` uses spectrum-level molecular annotations. Other fields use dataset
metadata. Every target is accompanied by an availability mask. Targets require
an annotation reader in each participating image context.

### Configure splitting

Split strategies support random splitting, single-label stratification,
mask-based selection, and grouped splitting. Fractions and seed are stored in
the dataset configuration. Group by `image_key` for cohort-level separation or
by annotation metadata fields when source groups must not cross partitions.

```python
partitions = wrapper.active_dataset.create_partitions()
train = partitions.train
validation = partitions.validation
test = partitions.test
manifest = partitions.manifest.get_config()
```

The manifest records stable sample identities, not only transient array indices.

### Configure a cohort dataset

Activate a cohort, then select the registered dataset:

```python
wrapper.cohorts.activate("kidney-cohort")
wrapper.models_manager.set_dataset(
    "CohortPixelDataset",
    normalization="tic",
    target_specs=target_specs,
    split={
        "strategy": "grouped",
        "seed": 0,
        "parameters": {"group_fields": "image_key"},
    },
)
```

All cohort members must expose consistent target schemas. Latent datasets also
require every member to have a latent reader.
