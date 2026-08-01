# Dataset management protocol

This guide describes how to turn reviewed external-database records into local imzML/ibd pairs, canonical molecular annotations, and a merged dataset.

## Scope

The guide covers query selections, METASPACE authentication, downloads, annotation retrieval, merge sampling, artifacts, and failure handling. 

> Interactive METASPACE discovery is covered by [07_metaspace_dataset_explorer_reworked_concise.ipynb notebook](../../assets/notebooks/tutorials/07_metaspace_dataset_explorer_reworked_concise.ipynb). 

> Internal component boundaries are described in [Dataset management internals](../library-internals/dataset-management.md).

## Required inputs

The download stage requires:

- a selection JSON exported by the query stage;
- a workspace path;
- annotation retrieval options;
- a METASPACE account and API key when the service requires authenticated download access;
- sufficient disk space for either all source pairs or one source pair plus the merged output.

The selection stores the source name, accepted dataset IDs, query filters, and reviewed metadata. Download consumes this snapshot and does not repeat dataset discovery.

## Create a METASPACE download session

Create or sign in to a METASPACE account at <https://metaspace2020.eu>. Generate an API key on the account page at <https://metaspace2020.eu/user/me>. Treat the key as a password. The [official annotation example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-dataset-annotations.html) also documents account-key authentication.

From the repository root, source the session script inside the environment used for the download:

```bash
source assets/scripts/datasets/metaspace_session.sh
```

The script:

1. reads the key without echoing it;
2. exports `METASPACE_API_KEY` in the current shell;
3. calls `metaspace_authentication.py` to verify `SMInstance.logged_in()`;
4. removes the variable if validation fails.

The key is not written to a project file. Commands launched from the authenticated shell inherit it. End the session with:

```bash
unset METASPACE_API_KEY
```

Do not execute the script as `bash assets/scripts/datasets/metaspace_session.sh`. An executed child shell cannot modify the parent environment. In a notebook, either launch Jupyter from an authenticated shell or source the script and run the download in the same `%%bash` cell.

METASPACE applies an account or service download quota. The [official client API reference](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html) documents the download methods but does not publish a stable numeric default. When the service returns `Download_Limit_Reached.txt`, the adapter raises `DownloadLimitError` before transferring files. Wait for the service quota to reset or contact METASPACE support; changing the local sampling settings does not change that server-side quota.

## Keep one annotation FDR

Use `annotation_fdr` for discovery statistics and annotation retrieval:

```json
{
  "annotation_fdr": 0.1,
  "include_spatial": true
}
```

The selection retains the `annotation_fdr` used by query. Download inherits this value. If annotation options request another value, validation stops the operation. The legacy retrieval key `fdr` is rejected because it does not state what the threshold controls.

Keep `include_spatial` enabled when merge selection depends on annotated pixels. Dataset-level molecular rows without ion images cannot establish molecule-to-pixel links.

## Download selected datasets

Materialize every record in a selection:

```bash
uv run python assets/scripts/datasets/manage_datasets.py download \
  --workspace-path workspace \
  --source metaspace \
  --selection workspace/datasets/selections/metaspace-selection.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json
```

Restrict a run without modifying the selection by repeating `--dataset-id`:

```bash
uv run python assets/scripts/datasets/manage_datasets.py download \
  --source metaspace \
  --selection workspace/datasets/selections/metaspace-selection.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json \
  --dataset-id DATASET_ID
```

Relative command-line paths are resolved against the repository root. The default workspace is `workspace`.

Before contacting METASPACE for a dataset, the adapter checks for non-empty `<dataset_id>.imzML` and `<dataset_id>.ibd` files in the destination directory. A complete local pair is reused. If one file is missing, only missing or empty files are transferred when signed links are available.

After each pair is present, the operation retrieves metadata and molecular results, retrieves first-isotope ion images, maps image coordinates to imzML spectrum IDs, and replaces that dataset's canonical SQLite annotations.

## Merge downloaded datasets

The `merge` command consumes local pairs declared in a merge configuration. Paths inside the configuration are relative to the configuration file:

```bash
uv run python assets/scripts/datasets/manage_datasets.py merge \
  --workspace-path workspace \
  --config assets/configs/datasets/merge.example.json
```

When an input omits `spectrum_ids`, merge includes every spectrum linked to at least one molecular annotation. It can add spectra without molecular links using:

- `unannotated_ratio`: requested count relative to the annotated count;
- `unannotated_amount`: requested absolute count;
- `random_seed`: reproducible sampling seed.

For each source dataset, the selected unannotated count is:

```text
min(max(floor(annotated_count * unannotated_ratio), unannotated_amount), available_unannotated_count)
```

Missing limits are treated as zero. If both limits are absent, only annotated spectra are merged. An explicit `spectrum_ids` list overrides this automatic selection for that input. A spectrum without a molecular link is not assumed to be biological background.

## Download and merge with bounded disk use

Use `download-merge` when source pairs should be processed one at a time:

```bash
uv run python assets/scripts/datasets/manage_datasets.py download-merge \
  --workspace-path workspace \
  --source metaspace \
  --selection workspace/datasets/selections/metaspace-selection.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json \
  --output workspace/datasets/merged/metaspace-selection/dataset.imzML \
  --merged-dataset-id metaspace-selection \
  --row-width 128 \
  --unannotated-ratio 1.0 \
  --random-seed 0
```

The command downloads one dataset into staging, stores its metadata and annotations, appends selected spectra, and removes its staging pair. Add `--keep-downloads` to retain source pairs under `workspace/datasets/sources/metaspace`.

## Verify the result

Expected artifacts are:

```text
workspace/datasets/
├── catalog.sqlite
├── selections/
├── sources/metaspace/<dataset_id>/
│   ├── <dataset_id>.imzML
│   └── <dataset_id>.ibd
└── merged/<merged_dataset_id>/
    ├── dataset.imzML
    └── dataset.ibd
```

Read annotations for one merged spectrum:

```python
from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader

reader = SQLiteAnnotationReader(
    "workspace/datasets/catalog.sqlite",
    merged_dataset_id="metaspace-selection",
)

source_metadata = reader.get_spectrum_metadata(0)
molecules = reader.get_spectrum_annotations(0)
```

`get_spectrum_annotations()` resolves the merged index to the source dataset and source spectrum ID before reading molecule links.

## Failures that require a decision

- `DownloadLimitError`: the service returned its quota sentinel; wait or contact METASPACE support.
- authentication failure: generate a current API key and source the script again.
- FDR mismatch: use the `annotation_fdr` stored in the selection.
- incomplete ion images: the requested molecular results cannot be mapped completely to pixels; do not treat the partial result as a labeled dataset.
- incomplete local pair: preserve the valid file and rerun download to retrieve the missing member.
- zero selected spectra: confirm that spatial annotations were requested and that the chosen FDR produced pixel links.
