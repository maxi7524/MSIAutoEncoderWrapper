# Persist and restore models

Model persistence stores configuration, weights, training history, context
configuration, and trained state as one artifact directory.

## Purpose and available operations

### Artifact identity

Artifacts are addressed by context/image name and model name. Models without an
image context use the `global` context key.

### Runtime binding

The model manager distinguishes the currently loaded model from a model bound to
one local image context. Loading another model does not overwrite an explicitly
bound local runtime.

## Detailed instructions

### Save the active model

```python
model_dir = wrapper.workspace.save_model(
    img_name="example_1",
    model_name="baseline",
    history=history,
)
```

Omitted names use the active context and active model. Saving requires a model
configuration and state dictionary. The artifact contains `config.json`,
`weights.pt`, and optional `history.json` under its configuration directory.

### Load weights and configuration

```python
model = wrapper.models_manager.load_model(
    img_name="example_1",
    model_name="baseline",
    strict=True,
    bind_to_local_context=False,
)
```

`strict=True` requires exact state-dictionary compatibility.
`bind_to_local_context=True` attaches the runtime to the active image context.

Restore a complete schema-v2 experiment directory with:

```python
config = wrapper.load_experiment("path/to/model-directory", strict=True)
```

### Attach an existing Torch model

```python
wrapper.models_manager.attach_model(
    torch_model=model,
    model_type="autoencoder",
    model_name="manual-ae",
    trained=True,
    bind_to_local_context=True,
)
```

`model_type` may be detected for known architectures. Set `trained=False` until
weights are ready for inference.

### Export a portable folder

```python
wrapper.workspace.export_model_folder(
    "exports/baseline",
    img_name="example_1",
    model_name="baseline",
    overwrite=False,
)
```

Export refuses to overwrite by default and copies the complete artifact rather
than a standalone weight file.
