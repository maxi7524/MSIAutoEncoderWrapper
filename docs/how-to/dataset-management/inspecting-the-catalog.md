# Inspect the dataset catalog

`DatasetCatalog` is the canonical SQLite store for source datasets, molecular
annotations, spectrum links, merged datasets, and source-index mappings.

## Purpose and available operations

### Source records

Source datasets are identified by `(source, dataset_id)` and retain metadata,
local path, and materialization status.

### Merged records

Merged datasets have an independent identifier and a mapping from merged index
to source spectrum identity.

## Detailed instructions

### Query datasets and paths

```python
from msi_autoencoder_wrapper.dataset_management.catalog import DatasetCatalog

catalog = DatasetCatalog("data/tutorial_workspace/datasets/catalog.sqlite")
datasets = catalog.list_datasets(source="metaspace", status="materialized")
record = catalog.get_dataset("metaspace", "dataset-id")
identity = catalog.resolve_dataset_path("path/to/image.imzML")
```

Both list filters are optional. Path resolution returns source or merged
identity when the file belongs to a registered local path.

### Query annotations and mappings

```python
annotations = catalog.get_annotations(
    source="metaspace",
    dataset_id="dataset-id",
    filters={"max_fdr": 0.1},
)
annotated_ids = catalog.get_annotated_spectrum_ids(
    source="metaspace",
    dataset_id="dataset-id",
)
source_index = catalog.get_source_index(
    merged_dataset_id="merged-id",
    merged_spectrum_index=0,
)
```

Catalog write methods replace annotations and mappings transactionally at the
dataset or merged-dataset boundary.
