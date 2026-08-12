# Inspect the dataset catalog

`DatasetCatalog` is the canonical SQLite store for source datasets, molecular
annotations, spectrum links, merged datasets, and source-index mappings. Table
responsibilities, transactions, and path identity are described in
[SQLite catalog internals](../../library-internals/dataset-management/sqlite-catalog.md).

## Purpose and available operations

### Two catalogs per cohort, different roles

`query` and `download` write a **working catalog** at
`workspace/configs/datasets/<cohort_id>/<cohort_id>.sqlite`. It tracks source
dataset identity and file-materialization status only; it holds no imported
annotations, because `download` writes annotations as CSV files instead (see
[Download selected datasets](downloading-datasets.md)).

`compose` writes a separate, self-contained **composed catalog** at
`workspace/datasets/<cohort_id>/<cohort_id>.sqlite`, colocated with the merged
image. This one holds the actually-imported annotations and merged-spectrum
provenance, and is what `msi_autoencoder_wrapper` reads automatically (see
[Compose a cohort dataset](composing-a-cohort.md)). Open the specific path
that was written by the operation whose data you want to read — the two are
never the same file.

### Source records

Source datasets are identified by `(source, dataset_id)` and retain metadata,
local path, and materialization status.

### Merged records

Merged datasets have an independent identifier and a mapping from merged index
to source spectrum identity.

## Detailed instructions

### Query datasets and paths

```python
from msi_dataset_manager.catalog import DatasetCatalog

catalog = DatasetCatalog("workspace/datasets/kidney/kidney.sqlite")  # composed catalog
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

`get_annotations()` also accepts `database_name`, `database_version`,
`formula`, and `adduct` filters. Catalog write methods
(`replace_annotations()`, `replace_spectrum_mappings()`) replace annotations
and mappings transactionally at the dataset or merged-dataset boundary, so
readers never observe a partially replaced generation. These methods only
return data from a **composed** catalog — a working catalog never has
annotation rows to query.

### Resolve merged provenance from either direction

```python
merged_index = catalog.get_merged_index(
    merged_dataset_id="merged-id",
    source="metaspace",
    source_dataset_id="dataset-id",
    source_spectrum_id=42,
)
contributing_sources = catalog.list_merged_sources("merged-id")
```

`get_merged_index()` is the inverse of `get_source_index()`. Prefer
`msi_autoencoder_wrapper.annotations.SQLiteAnnotationReader` over these raw
catalog methods for reading merged spectra during model training or analysis;
it wraps them with the provider-independent reader contract used elsewhere in
the wrapper. See [Retrieve dataset annotations](retrieving-annotations.md#read-normalized-annotations).
