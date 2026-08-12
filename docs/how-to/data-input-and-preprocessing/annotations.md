# Attach molecular annotations

This guide configures molecular annotations for a single-image context and
documents every supported annotation-source selection path.

## Purpose and available operations

### Annotation responsibilities

An MSI data reader provides spectra and spatial coordinates. An annotation
reader is a separate component that provides dataset metadata, molecular rows,
and molecule-to-spectrum links. Attaching annotations does not change the
spectra returned by the data reader.

The configured annotation reader is available as
`wrapper.active_context.annotation_reader`. It supports:

- reading metadata for the selected source dataset;
- reading all molecular annotations, optionally with read-time filters;
- reading molecules associated with one zero-based spectrum ID;
- reading source metadata associated with one spectrum;
- exporting its constructor configuration through the shared configurable
  component interface.

### Supported sources and selection priority

The library supports paired local annotation CSV exports and the canonical
SQLite catalog, in one of two locations. Automatic selection uses this order:

1. Find a local annotation CSV pair beside the imzML file: canonical
   `annotations.csv` + `pixel_intensities.csv` first (the files
   `msi-datasets download` writes — see
   [Download selected datasets](../dataset-management/downloading-datasets.md)),
   or, if either canonical file is absent, the legacy pair
   `metaspace_annotations.csv` + `<imzML-stem>_pixel_intensities.csv` (or
   exactly one other `*_pixel_intensities.csv` file, if unambiguous).
2. If no local CSV pair is complete, resolve the image in a same-named SQLite
   catalog beside the imzML file — `<imzML-stem>.sqlite` — the **composed
   catalog** that `msi-datasets compose` writes next to its merged output; see
   [Compose a cohort dataset](../dataset-management/composing-a-cohort.md).
3. If that sibling catalog does not exist or does not register the image,
   fall back to the legacy single workspace catalog at
   `<project_path>/datasets/catalog.sqlite`.
4. If none of the above resolves, leave the annotation reader unset and log a
   warning.

A local CSV pair wins over any SQLite catalog when both are present. Passing
`annotation_catalog_path` makes that specific catalog mandatory and skips CSV
detection entirely: the library does not fall back to CSV, the sibling
catalog, or the legacy catalog when the explicit file is missing or does not
contain the image.

## Detailed instructions

### Configure annotations with the MSI reader

`ContextManagerProxy.set_reader()` configures spectra first and, by default,
runs annotation-source detection for the same image context.

```python
from msi_autoencoder_wrapper import MSIAutoEncoderWrapper

wrapper = MSIAutoEncoderWrapper(project_path="data/tutorial_workspace")
reader = wrapper.context_manager.set_reader(
    reader_name_or_instance="PyImzMLReader",
    img_name_or_path=(
        "data/tutorial_workspace/datasets/example_1/example_1.imzML"
    ),
    annotation_catalog_path=None,
    auto_load_annotations=True,
)
```

#### Parameters of `set_reader()` that affect annotations

- `reader_name_or_instance` accepts a registered reader key, compatible reader
  class, or initialized reader instance. String and class targets receive the
  remaining keyword arguments as constructor parameters.
- `img_name_or_path` selects the image context. It can be an image known to the
  workspace or an existing image path.
- `annotation_catalog_path` selects an explicit SQLite catalog, skipping
  automatic detection entirely. `None` runs the full automatic-detection order
  described above (local CSV pair, then sibling composed catalog, then the
  legacy workspace catalog).
- `auto_load_annotations=True` runs that detection after the data reader
  is configured.
- `auto_load_annotations=False` configures spectra without touching the
  annotation reader. This is required when annotations will be attached later
  or supplied by a custom implementation.
- additional keyword arguments configure the selected MSI data reader; they are
  not forwarded to the annotation reader.

The returned value is the configured MSI data reader, not the annotation
reader. Inspect the active context to obtain the automatically selected
annotation reader.

### Use an SQLite catalog

A catalog must map the configured imzML path to either one source dataset or
one merged dataset. Automatic resolution tries, in order, the composed catalog
`<imzML-stem>.sqlite` beside the image — the file
[`msi-datasets compose`](../dataset-management/composing-a-cohort.md) writes
next to its merged output — then the legacy single workspace catalog at
`<project_path>/datasets/catalog.sqlite`. Whichever one registers the image
supplies the corresponding SQLite reader parameters:

- source dataset: `catalog_path`, `source`, and `dataset_id`;
- merged dataset: `catalog_path` and `merged_dataset_id`.

Use another catalog explicitly when the image registration is outside both of
those locations:

```python
wrapper.context_manager.set_reader(
    reader_name_or_instance="PyImzMLReader",
    img_name_or_path="path/to/image.imzML",
    annotation_catalog_path="path/to/catalog.sqlite",
)
```

#### Configure `SQLiteAnnotationReader` directly

For an unmerged source dataset, `source` and `dataset_id` must be supplied
together:

```python
from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader

annotation_reader = SQLiteAnnotationReader(
    catalog_path="data/tutorial_workspace/datasets/catalog.sqlite",
    source="metaspace",
    dataset_id="dataset-id",
    merged_dataset_id=None,
    default_filters={"max_fdr": 0.1},
)

wrapper.context_manager.set_annotation_reader(
    reader_name_or_instance=annotation_reader,
    img_name_or_path="path/to/image.imzML",
)
```

For a merged dataset, omit `source` and `dataset_id` and select exactly one
`merged_dataset_id`:

```python
annotation_reader = SQLiteAnnotationReader(
    catalog_path="data/tutorial_workspace/datasets/catalog.sqlite",
    merged_dataset_id="merged-dataset-id",
)
```

`default_filters` is an optional mapping applied to every read. SQLite supports
`database_name`, `database_version`, `formula`, `adduct`, and `max_fdr`.
Filters passed to a read method override default values with the same keys.

### Use paired local annotation CSV files

Automatic CSV detection first looks for the canonical pair
`annotations.csv` + `pixel_intensities.csv` beside the imzML/ibd pair — the
files [`msi-datasets download`](../dataset-management/downloading-datasets.md)
writes:

```text
example_1/
├── example_1.imzML
├── example_1.ibd
├── annotations.csv
└── pixel_intensities.csv
```

If either canonical file is absent, it falls back to the legacy METASPACE
export names: `metaspace_annotations.csv` plus one intensity table. The
preferred legacy intensity filename is `<imzML-stem>_pixel_intensities.csv`;
if that is absent, exactly one other `*_pixel_intensities.csv` file is
accepted. Multiple fallback candidates raise `ValidationError` because the
intended image association cannot be inferred.

#### Configure `MetaspaceCSVAnnotationReader` directly

Direct configuration supports arbitrary filenames and does not depend on
automatic naming:

```python
from msi_autoencoder_wrapper.annotations import MetaspaceCSVAnnotationReader

annotation_reader = MetaspaceCSVAnnotationReader(
    image_path="path/to/image.imzML",
    annotations_path="path/to/annotations.csv",
    pixel_intensities_path="path/to/intensities.csv",
    default_filters={"max_fdr": 0.1, "adduct": "+H"},
)

wrapper.context_manager.set_annotation_reader(
    reader_name_or_instance=annotation_reader,
    img_name_or_path="path/to/image.imzML",
)
```

Constructor parameters:

- `image_path` identifies the imzML file used to map `x*_y*` columns to
  zero-based spectrum IDs; a sibling `.ibd` file is required;
- `annotations_path` identifies the METASPACE molecular annotation table;
- `pixel_intensities_path` identifies the table containing `mz` and spatial
  intensity columns;
- `default_filters` defines filters applied to every read;
- `active_context` is injected by the context manager and normally should not
  be set by user code.

CSV reads support `max_fdr` and equality filters for retained record fields,
including `formula` and `adduct`. A filter supplied to `get_annotations()` or
`get_spectrum_annotations()` overrides a default filter with the same name.

### Attach annotations after reader setup

Disable automatic loading when configuring spectra, then call
`set_annotation_reader()` without a strategy to run detection later:

```python
wrapper.context_manager.set_reader(
    "PyImzMLReader",
    "path/to/image.imzML",
    auto_load_annotations=False,
)

annotation_reader = wrapper.context_manager.set_annotation_reader(
    reader_name_or_instance=None,
    img_name_or_path="path/to/image.imzML",
)
```

Parameters of `set_annotation_reader()`:

- `reader_name_or_instance=None` requests automatic SQLite/CSV detection;
- a registered key, compatible class, or initialized instance selects a reader
  explicitly;
- `img_name_or_path` selects the receiving image context;
- additional keyword arguments are passed to an explicitly selected reader
  constructor;
- `catalog_path` is consumed as the catalog override when automatic detection
  is requested.

The method returns the selected annotation reader or `None` when automatic
detection finds no source.

### Read and verify annotations

Activate the image before reading through `active_context`:

```python
wrapper.workspace.set_active_image("path/to/image.imzML")
annotation_reader = wrapper.active_context.annotation_reader

dataset_metadata = annotation_reader.get_dataset_metadata()
all_annotations = annotation_reader.get_annotations()
filtered_annotations = annotation_reader.get_annotations(
    {"max_fdr": 0.05, "adduct": "+H"}
)
spectrum_annotations = annotation_reader.get_spectrum_annotations(
    spectrum_id=0,
    filters={"max_fdr": 0.05},
)
spectrum_metadata = annotation_reader.get_spectrum_metadata(0)
```

`get_dataset_metadata()` returns one source record for an unmerged SQLite
dataset, a merged-source summary for a merged dataset, or metadata retained in
the METASPACE CSV export. `get_spectrum_annotations()` uses an indexed SQLite
lookup for catalog data and normalized `spectrum_ids` for CSV data.

For CSV records, positive values are stored in `spectrum_values` and their keys
are listed in `spectrum_ids`. Zero or empty intensities do not create a molecular
link. Negative, infinite, and NaN intensities raise `ValidationError`.

### Handle validation failures

The following conditions require correction rather than automatic fallback:

- an explicit catalog path does not exist;
- the image is absent from an explicitly selected catalog;
- both source-dataset and merged-dataset selectors are supplied to the SQLite
  reader, or neither selector is supplied;
- only one of `source` and `dataset_id` is supplied;
- the imzML/ibd pair or either CSV file is missing;
- the annotation CSV contains no records or a record without `mz`;
- pixel-intensity `mz` values are duplicated or do not match annotation rows;
- a pixel intensity is negative or non-finite;
- several fallback intensity CSV files make automatic selection ambiguous.

If automatic detection finds no catalog registration and no complete CSV pair,
it logs a warning and leaves annotations unset. Spectrum reading remains
available in that context.
