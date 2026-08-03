# Work with image contexts

An image context groups the reader, annotation reader, binner, inverse binner,
normalization pipeline, and optional local model for one MSI image.

## Purpose and available operations

### Context locality

Every image has a separate configuration ledger entry. Changing the active
image changes which local components are exposed through `active_context`; it
does not overwrite another image's components.

### Context operations

The context manager can set or inspect components, export the current context
configuration, and restore a context from configuration. The workspace controls
which image is active.

## Detailed instructions

### Activate an image

```python
image = "data/tutorial_workspace/datasets/example_1/example_1.imzML"
wrapper.workspace.set_active_image(image)
```

`set_active_image()` accepts a known workspace image name or an existing path.
Use `wrapper.workspace.use_image(image)` as a context manager when activation
must be temporary:

```python
with wrapper.workspace.use_image(image):
    reader = wrapper.active_context.reader
```

### Configure local components

```python
wrapper.context_manager.set_reader("PyImzMLReader", image)
wrapper.context_manager.set_binner("LinearBinning", image, bin_step=0.1)
wrapper.context_manager.set_inverse_binner(
    "TopPeaksInverseBinner",
    image,
    top_k=100,
)
wrapper.context_manager.set_normalization(
    {
        "stage": "binned",
        "steps": {"tic": {"type": "scalar", "kind": "tic"}},
        "reconstruction": {
            "output_space": "source",
            "denormalization_stage": "after_inverse_binning",
        },
    },
    image,
)
```

The active reader supplies `x_min` and `x_max` when `LinearBinning` does not
receive them explicitly. The active forward binner is injected into an inverse
binner when its `binner` parameter is omitted.

### Inspect and restore configuration

```python
config = wrapper.context_manager.get_context_config("example_1")
restored = wrapper.context_manager.load_context_config(
    config,
    img_name_or_path=image,
    base_path=wrapper.project_path,
)
```

`base_path` resolves relative file paths stored in portable configuration. The
returned mapping contains the components restored for the image.

### Clear active state

```python
wrapper.active_context.clear_active_context()
wrapper.workspace.clear_active_context()
```

Clearing active state removes runtime selections. It does not delete dataset or
model files.
