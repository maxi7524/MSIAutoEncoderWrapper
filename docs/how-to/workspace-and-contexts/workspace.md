# Configure a workspace

The workspace stores datasets, model configurations, weights, histories, latent
representations, and cohort definitions under one project root.

## Purpose and available operations

### Workspace ownership

`MSIAutoEncoderWrapper(project_path=...)` creates a workspace proxy for the
selected root. The proxy resolves paths; it does not copy external images into
the workspace automatically.

The tutorial workspace is `data/tutorial_workspace`. Its datasets are stored in
`data/tutorial_workspace/datasets`.

### Managed artifacts

The workspace can create required directories, select images and models, scan
saved models, save complete model artifacts, load configurations and weights,
and export one model folder.

## Detailed instructions

### Create or open the workspace

```python
from msi_autoencoder_wrapper import MSIAutoEncoderWrapper

wrapper = MSIAutoEncoderWrapper(
    project_path="data/tutorial_workspace",
    device="cpu",
)
wrapper.workspace.create_required_directories()
```

`project_path` accepts a string or path. `device` selects the Torch execution
device used by model and preprocessing operations. The wrapper creates a missing
project root after validating that its closest existing parent is writable.

### Resolve workspace paths

```python
datasets = wrapper.workspace.get_datasets_dir()
catalog = wrapper.workspace.get_dataset_catalog_path()
models = wrapper.workspace.get_models_root()
wrapper.workspace.print_workspace_layout()
```

Model paths require an image/context name and model name:

```python
model_dir = wrapper.workspace.get_model_dir("example_1", "baseline")
config_dir = wrapper.workspace.get_config_dir("example_1", "baseline")
latent_dir = wrapper.workspace.get_latent_dir("example_1", "baseline")
```

### Scan and export model artifacts

```python
available = wrapper.workspace.scan_available_models("example_1")
exported = wrapper.workspace.export_model_folder(
    destination="exports/example_1-baseline",
    img_name="example_1",
    model_name="baseline",
    overwrite=False,
)
```

`overwrite=False` rejects an existing destination. Export copies the complete
model artifact folder so configuration, weights, and history remain together.

### Handle workspace validation

Use an explicit writable project path. Model operations require both a selected
image context and model name. Missing configuration, weights, or context paths
raise project exceptions instead of returning partially initialized models.
