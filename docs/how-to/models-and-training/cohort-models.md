# Configure cohort models

Cohort models consume a multi-image dataset while retaining image and local
spectrum identity for every sample.

## Purpose and available operations

### Training contexts

A model may be trained over `CohortPixelDataset` using original spectra or over
`CohortLatentDataset` using materialized embeddings. This is a cohort
(multi-image) context, not a separate “global model” architecture family.

### Existing-model policies

Cohorts may also reference one shared autoencoder or per-image autoencoders for
transforming members.

## Detailed instructions

### Train one model on multiple images

```python
wrapper.cohorts.activate("kidney-cohort")
wrapper.models_manager.set_dataset(
    "CohortPixelDataset",
    split={
        "strategy": "grouped",
        "seed": 0,
        "parameters": {"group_fields": "image_key"},
    },
)
wrapper.models_manager.set_model_type("autoencoder", "cohort-ae")
# Configure components, compile, and train as for a single-image model.
```

Group by `image_key` when complete images must remain in one partition. Random
pixel splitting does not measure cross-image generalization.

### Preserve sample identity

`CohortDataset.get_sample_id()` returns `image_key` and local `spectrum_id`.
These identities are used by split manifests and downstream spatial mapping.

### Use latent cohort input

```python
wrapper.models_manager.set_dataset(
    "CohortLatentDataset",
    normalization="none",
)
```

Every member must have a materialized latent reader with compatible feature
shape. Missing latent readers or inconsistent target schemas raise validation
errors before training.
