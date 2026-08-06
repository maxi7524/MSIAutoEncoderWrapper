# Import a local METASPACE dataset

Local import adds one imzML/ibd pair and paired METASPACE CSV exports to the
canonical SQLite catalog. The shared CSV-to-canonical normalization is
described in
[Annotation normalization internals](../../library-internals/dataset-management/annotation-normalization.md).

## Purpose and available operations

### Input files

Import requires the image pair, molecular annotation CSV, and per-pixel
intensity CSV. It maps spatial CSV columns to zero-based imzML spectrum IDs.

### Shared normalization

The importer and `MetaspaceCSVAnnotationReader` use the same CSV normalization
function, so direct reading and catalog import produce the same molecular links.

## Detailed instructions

### Import through Python

```python
from msi_autoencoder_wrapper.dataset_management.catalog import DatasetCatalog
from msi_autoencoder_wrapper.dataset_management.operations import import_local_dataset

catalog = DatasetCatalog("data/tutorial_workspace/datasets/catalog.sqlite")
result = import_local_dataset(
    catalog=catalog,
    source="metaspace",
    dataset_id="example-1",
    name="Example 1",
    imzml_path="data/tutorial_workspace/datasets/example_1/example_1.imzML",
    annotations_path="data/tutorial_workspace/datasets/example_1/metaspace_annotations.csv",
    pixel_intensities_path="data/tutorial_workspace/datasets/example_1/example_1_pixel_intensities.csv",
    metadata={},
)
```

The result contains `spectra`, `annotations`, and `spatial_links` counts.

### Validate source values

Mass-to-charge values must match uniquely between annotation and intensity
tables. Negative or non-finite intensities are rejected. Zero and empty values
do not create spatial links.
