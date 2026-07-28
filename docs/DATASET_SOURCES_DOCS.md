# Dataset Management Tutorial and Configuration Reference

This document explains how external MSI datasets are queried, downloaded,
normalized, merged, and loaded for training. METASPACE is the first supported
database. Database-specific behavior is isolated in `dataset_sources`; every
source must produce the same local SQLite annotation format.

## Concepts and architecture

The pipeline has two independent responsibilities:

```text
External database
        |
        v
dataset_sources
  query, source filters, download, source parsing, normalization
        |
        v
Canonical workspace dataset
  imzML + ibd + datasets/catalog.sqlite
        |
        +-- readers: spectrum access by spectrum_id
        |
        +-- annotations: canonical SQLite metadata and annotation access
                          |
                          v
                     PixelDataset
                     target selection and class assignment
```

`dataset_sources` is responsible for external databases. It contains the
METASPACE implementation and will contain future database strategies. Each
strategy retrieves complete source metadata and molecular annotations and
normalizes them before they are stored locally.

The bundled PRIDE strategy treats every complete ``.imzML``/``.ibd`` pair as
one image dataset, even when one PRIDE project contains several acquisitions.
It accepts only an explicitly associated ``<image>_annotations.csv`` or
``<image>_annotations.tsv`` table containing a molecule identity and
``spectrum_id``. It never infers molecule identities from m/z peaks. Project
records with missing or ambiguous required biological metadata, incomplete
pairs, unsupported annotation files, or missing required checksums are logged
with a link to their PRIDE project and excluded from the selection.

`annotations` does not contain METASPACE parsing. `SQLiteAnnotationReader`
reads the single canonical SQLite representation produced by
`dataset_sources`. It joins molecular annotations to the same zero-based
`spectrum_id` used by the imzML readers.

`PixelDataset` is the first layer that assigns model class indices. Downloaded
metadata remains semantic data such as `condition = "disease"`; it is not
changed to class zero during import.

## Workspace layout

The default workspace is `<repository_root>/workspace`:

```text
workspace/
└── datasets/
    ├── catalog.sqlite
    ├── selections/
    │   └── pilot.json
    ├── sources/
    │   └── metaspace/
    │       └── <dataset_id>/
    │           ├── <dataset_id>.imzML
    │           └── <dataset_id>.ibd
    └── merged/
        └── <merged_dataset_id>/
            ├── dataset.imzML
            └── dataset.ibd
```

Metadata and annotations are stored in `catalog.sqlite`. They are not copied to
large JSON files beside every imzML pair. A small JSON selection file is still
used between Query and Download. It contains only the reproducible list of
selected external dataset IDs and their query metadata; it is not the
annotation database.

## Usage

### CLI tutorial

Run commands from any directory. Relative command-line paths are resolved
against the repository root, not the shell's current working directory. The CLI
prints the resolved repository, workspace, catalog, configuration, and output
paths before performing work.

`--workspace-path` is optional and defaults to `workspace`, which means
`<repository_root>/workspace`:

```bash
python assets/scripts/datasets/manage_datasets.py --help
```

#### Step 1: Query

Query searches the external database and stores complete dataset metadata in
SQLite. It does not download imzML or ibd files.

```bash
python assets/scripts/datasets/manage_datasets.py query \
  --source metaspace \
  --filters assets/configs/datasets/metaspace_filters.json \
  --selection workspace/datasets/selections/pilot.json
```

To use another workspace:

```bash
python assets/scripts/datasets/manage_datasets.py query \
  --workspace-path experiments/pilot_workspace \
  --source metaspace \
  --filters assets/configs/datasets/metaspace_filters.json \
  --selection experiments/pilot_workspace/datasets/selections/pilot.json
```

Review the selection file and the SQLite catalog before downloading. For the
first functional test, keep only three or four dataset IDs.

#### Step 2: Download

Download materializes the selected imzML/ibd pairs and imports all available
metadata and molecular annotations into SQLite:

```bash
python assets/scripts/datasets/manage_datasets.py download \
  --source metaspace \
  --selection workspace/datasets/selections/pilot.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json
```

Download one selected dataset without modifying the selection file:

```bash
python assets/scripts/datasets/manage_datasets.py download \
  --source metaspace \
  --selection workspace/datasets/selections/pilot.json \
  --dataset-id <dataset_id>
```

Repeat `--dataset-id` to download several explicit datasets. If no dataset ID is
provided, every dataset in the selection is downloaded.

#### Step 3: Merge

Merge already downloaded local imzML pairs using a merge configuration:

```bash
python assets/scripts/datasets/manage_datasets.py merge \
  --config assets/configs/datasets/merge.example.json
```

Paths declared inside a configuration file are resolved relative to the
directory containing that configuration. This differs intentionally from CLI
arguments, which are repository-relative. A portable configuration can
therefore use paths such as `../../workspace/datasets/sources/...`.

The merged image uses consecutive rectangular coordinates. SQLite preserves
the stable mapping:

```text
(source, source_dataset_id, source spectrum_id) -> merged spectrum_id
```

This is the provenance needed to recover source annotations after merging.
Spatial coordinates remain owned by the imzML readers and are not duplicated in
the annotation database.

#### Low-disk download and merge

Use `download-merge` when all source imzML pairs should not coexist on disk:

```bash
python assets/scripts/datasets/manage_datasets.py download-merge \
  --source metaspace \
  --selection workspace/datasets/selections/pilot.json \
  --output workspace/datasets/merged/pilot/dataset.imzML \
  --merged-dataset-id pilot \
  --row-width 128
```

This operation downloads one source dataset, imports its metadata and
annotations, appends its selected spectra, and removes the temporary source
pair. SQLite and the final merged imzML/ibd pair remain. Add `--keep-downloads`
only if the original source files must also remain in the workspace.

## Configuration explanation

### Query filter configuration

`metaspace_filters.json` is a JSON object passed directly to the official
METASPACE `SMInstance.datasets(...)` method. Example:

```json
{
  "organism": "Homo sapiens",
  "polarity": "Positive"
}
```

These are database query filters. They decide which external dataset records
enter the selection. Unsupported keys are allowed to fail visibly in the
METASPACE client; the library does not silently reinterpret them.

Do not use query filters to define final training classes. During the initial
functional test, query broadly enough to obtain three or four suitable images,
then inspect their metadata before creating the train/validation/test split.

### Annotation retrieval configuration

`metaspace_annotations.json` controls data retrieval, not training-time target
selection:

```json
{
  "fdr": 0.5,
  "include_spatial": true
}
```

Supported fields are:

- `fdr`: maximum FDR requested from METASPACE. The default is `0.5`, the
  broadest standard result level used by the integration.
- `databases`: optional list of METASPACE database selectors. If omitted, all
  databases attached to the source dataset are queried.
- `include_spatial`: whether first-isotope ion images are downloaded and mapped
  to `spectrum_id`. The default is `true`.

The initial pipeline stores all returned molecules. No formula, adduct,
database, or stricter FDR filter is applied by default.

### Merge configuration

The merge configuration selects local inputs and optionally selected spectrum
IDs:

```json
{
  "merged_dataset_id": "pilot",
  "output_path": "../../../workspace/datasets/merged/pilot/dataset.imzML",
  "row_width": 128,
  "inputs": [
    {
      "source": "metaspace",
      "dataset_id": "<dataset_id>",
      "imzml_path": "../../../workspace/datasets/sources/metaspace/<dataset_id>/<dataset_id>.imzML",
      "spectrum_ids": [0, 1, 2]
    }
  ]
}
```

`spectrum_ids` contains reader spectrum IDs, not coordinate tuples.

## Canonical SQLite format

The important logical tables are:

- `datasets`: source identity, complete metadata, local lifecycle state;
- `annotations`: canonical molecule identity, database provenance, FDR, and the
  complete normalized source record;
- `spectrum_annotations`: sparse many-to-many relation between `spectrum_id`
  and molecular annotations, with optional ion-image intensity;
- `merged_datasets`: locally produced merged imzML datasets;
- `spectrum_mappings`: reversible source-to-merged spectrum mapping.

One spectrum can contain multiple molecular annotations. One molecular
annotation can occur at multiple spectra. SQLite indexes this relation without
duplicating the complete molecular record for every pixel.

Complete source metadata is retained. Fields such as `condition`, `disease`,
`patient_id`, tissue, acquisition settings, and database details are not
converted to model classes by `dataset_sources` or `annotations`. If future
databases use different field names, normalization will be extended in their
source strategies while preserving the canonical reader API.

## Loading annotations

For one downloaded source dataset:

```python
from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader

annotation_reader = SQLiteAnnotationReader(
    "workspace/datasets/catalog.sqlite",
    source="metaspace",
    dataset_id="<dataset_id>",
)

metadata = annotation_reader.get_dataset_metadata()
all_molecules = annotation_reader.get_annotations()
pixel_molecules = annotation_reader.get_spectrum_annotations(spectrum_id=0)
```

For a merged dataset:

```python
annotation_reader = SQLiteAnnotationReader(
    "workspace/datasets/catalog.sqlite",
    merged_dataset_id="pilot",
)
```

The merged reader maps each merged `spectrum_id` back to its source dataset and
source `spectrum_id` before loading metadata and molecular annotations.

## Filtering

Filtering happens at two separate times.

### External database filtering

Query filters reduce the list of external datasets before download. They use
the source database's native fields and semantics. For METASPACE they are passed
to `SMInstance.datasets(...)`.

### Local annotation filtering

After download, the following optional molecular filters can be applied by the
SQLite reader without contacting METASPACE again:

- `database_name`: exact database name;
- `database_version`: exact database version;
- `formula`: exact molecular formula;
- `adduct`: exact adduct;
- `max_fdr`: maximum stored FDR.

Example:

```python
selected = annotation_reader.get_annotations(
    {
        "database_name": "HMDB",
        "max_fdr": 0.1,
    }
)
```

With no filters, all imported molecular annotations are returned. The initial
functional test should use this default. More restrictive training filters
should be designed only after download, loading, merging, and training have
been verified on the small pilot selection.

## Creating model targets in PixelDataset

Target fields are selected in `PixelDataset`, not during database import:

```python
dataset = PixelDataset(
    active_context=active_context,
    target_fields=["condition", "molecule"],
)
```

When `target_fields` is omitted, the existing sample contract remains:

```text
(spectrum_id, spectrum_tensor)
```

When targets are requested, a sample contains:

```text
(spectrum_id, spectrum_tensor, targets)
```

`condition` and other metadata fields produce one integer class target.
`molecule` produces a multi-label vector because one spectrum may contain many
molecules. Generated class maps are reproducible: unique semantic values are
sorted alphabetically. The maps can be inspected with:

```python
dataset.get_class_mappings()
```

For experiment-to-experiment stability, provide explicit maps and save them
with the experiment configuration:

```python
dataset = PixelDataset(
    active_context=active_context,
    target_fields=["condition"],
    class_mappings={
        "condition": {
            "control": 0,
            "disease": 1
        }
    },
)
```

No implicit background, unknown, or class-zero label is added. That policy and
noise/background detection require a separate filtering decision and are not
part of the initial pipeline test.

## Validation and troubleshooting

Validation is separated by responsibility:

- configuration validators check query selections and required configuration
  fields;
- source-data validators check records returned by database strategies;
- canonical-dataset validators check downloaded imzML/ibd pairs;
- annotation validators check the canonical annotation store and records.

If files appear in an unexpected directory, inspect the resolved paths printed
by the CLI. Remember:

- relative CLI paths are repository-relative;
- relative paths inside a configuration are configuration-file-relative;
- absolute paths are used unchanged;
- the default workspace is `<repository_root>/workspace`.

If dataset CLI startup is slow, verify that code imports
`msi_autoencoder_wrapper.dataset_sources`. The package root and METASPACE client
are loaded lazily, so querying datasets does not initialize model,
architecture, criterion, or training discovery.

## Recommended initial verification

1. Query METASPACE and retain three or four dataset IDs.
2. Download one dataset and inspect SQLite metadata and annotation counts.
3. Confirm that imzML spectra and SQLite annotations share `spectrum_id`.
4. Download and merge the remaining pilot datasets.
5. Build `PixelDataset` without filters and inspect several target samples.
6. Train one small reconstruction model before enabling condition or molecule
   heads.
7. Preserve dataset-level grouping metadata when creating train, validation,
   and test splits; do not split related pixels independently.
