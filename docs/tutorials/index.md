# Tutorials

The tutorials are executable, result-oriented introductions to the library.
They use the repository's `data/tutorial_workspace` workspace and its
`example_1` and `example_2` datasets. Detailed parameter references and
alternative configurations remain in the [how-to guides](../how-to/index.md).

## Prepare the tutorial workspace

Run all notebooks from the repository root. The example datasets are expected
under `data/tutorial_workspace/datasets`. If they are absent, run:

```bash
python docs/tutorials/download_tutorial_data.py
```

The download source is intentionally marked as TODO until the maintainer adds
the public Google Drive bundle. Existing local datasets are never replaced.

## Tutorial groups

```{toctree}
:maxdepth: 2

workspace-and-contexts/index
autoencoder/index
cohort-models/index
dataset-management/index
cli-and-configuration/index
```
