# Discover external datasets

Dataset discovery queries a registered provider and returns accepted and
rejected records without downloading MSI files.

## Purpose and available operations

### Providers

The built-in provider strategies are METASPACE and PRIDE. Each implements the
shared dataset-source contract but exposes provider-specific filter keys and
download behavior.

### Discovery outputs

`DatasetExplorer` retains active filters, accepted records, rejected records,
and source diagnostics so the user can review results before materialization.

## Detailed instructions

### Explore through Python

```python
from msi_autoencoder_wrapper.dataset_management.exploration import DatasetExplorer

explorer = DatasetExplorer(source="metaspace")
available = explorer.get_available_filters()
values = explorer.get_available_values("organism")
explorer.set_filters({"organism": "Mus musculus"})
records = explorer.search()
rejected = explorer.rejected()
```

Use `available_filters()` as an alias returning filter metadata. Filter keys and
values are validated by the selected provider.

### Query through the dataset CLI

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py query \
  --source metaspace \
  --workspace-path data/tutorial_workspace \
  --filters assets/configs/datasets/metaspace_filters.json \
  --selection data/tutorial_workspace/datasets/selections/query.json
```

The query writes discovery records into the catalog and exports a selection
snapshot. It does not download image pairs.
