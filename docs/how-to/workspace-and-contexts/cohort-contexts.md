# Work with cohort contexts

A cohort context groups configured image contexts for multi-image datasets and
model execution without repeatedly changing global workspace state.

## Purpose and available operations

### Cohort membership

Each cohort member retains an image key, reader, binner, annotation reader, and
optional latent reader. Membership is immutable at the context-object level;
manager operations replace the stored context with an updated value.

### Model policies

A cohort can reference one shared autoencoder or one autoencoder per image. The
references may point to model artifacts rather than keeping every model loaded.

## Detailed instructions

### Create and populate a cohort

Configure every image locally before adding it:

```python
cohort = wrapper.cohorts.create("kidney-cohort")
cohort = wrapper.cohorts.set_images(
    ["example_1", "example_2"],
    name="kidney-cohort",
)
cohort = wrapper.cohorts.add_image("another-image", name="kidney-cohort")
cohort = wrapper.cohorts.remove_image("another-image", name="kidney-cohort")
```

Image keys must resolve to configured local contexts. Duplicate members are not
added twice.

### Attach latent representations

```python
cohort = wrapper.cohorts.set_latent(
    image_key="example_1",
    path="path/to/example_1-latent.imzML",
    name="kidney-cohort",
)
```

`CohortLatentDataset` requires a latent reader for every member. Missing latent
representations are reported together when the dataset is constructed.

### Select the autoencoder policy

Use one model for all members:

```python
wrapper.cohorts.set_autoencoder(
    policy="common",
    model="models/shared-autoencoder",
    name="kidney-cohort",
)
```

Use separate references keyed by image:

```python
wrapper.cohorts.set_autoencoder(
    policy="per_member",
    models={
        "example_1": "models/example-1-autoencoder",
        "example_2": "models/example-2-autoencoder",
    },
    name="kidney-cohort",
)
```

`policy` must be `common` or `per_member`. Common mode requires `model`;
per-member mode requires a reference for every member.

### Activate, save, and restore the cohort

```python
wrapper.cohorts.activate("kidney-cohort")
path = wrapper.cohorts.save("kidney-cohort")
wrapper.cohorts.deactivate()

restored = wrapper.cohorts.load_config(
    cohort.get_config(),
    activate=True,
    base_path=wrapper.project_path,
)
```

Activation exposes the cohort to cohort datasets. It does not replace the
active single-image context.
