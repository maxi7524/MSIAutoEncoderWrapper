# Download selected datasets

Materialization downloads imzML/ibd files for accepted selection records and
updates their canonical catalog entries. The METASPACE source-download and
reuse flow is described in
[METASPACE provider internals](../../library-internals/dataset-management/metaspace-provider.md).

## Purpose and available operations

### Required inputs

Download requires a source strategy, selection JSON, workspace dataset
directory, catalog, optional annotation settings, and provider authentication
when required.

### Reuse behavior

A selection is processed one dataset at a time. Before requesting source files,
the operation checks
`workspace/datasets/sources/<source>/<dataset_id>/`. A complete non-empty local
imzML/ibd pair bypasses the provider download operation. If the pair is
incomplete, only that dataset is passed to the provider; the official
METASPACE client can skip an existing member and transfer its missing
companion. The wrapper validates the complete pair afterward.

For METASPACE selections, the wrapper validates that the provider exposes an
imzML/ibd pair and delegates the transfer to the official
`SMDataset.download_to_dir()` method. After the client returns, both local
files must exist before the catalogue is updated to materialized state.

This reuse behavior applies to the source imzML/ibd files, not to annotation
ion images. Ion images are retrieved through `all_annotation_images()` and are
not stored as individual image files. Repeating spatial annotation retrieval
requests the qualifying images again; an interrupted request cannot resume
from a partially downloaded image set.

### Output names and configuration

The source filename comes from the stable `dataset_id` stored in the selection,
not from the dataset display name or annotation configuration. For a workspace
at `data/tutorial_workspace`, dataset `2026-07-27_08h49m39s` is stored as:

```text
data/tutorial_workspace/datasets/sources/metaspace/
└── 2026-07-27_08h49m39s/
    ├── 2026-07-27_08h49m39s.imzML
    └── 2026-07-27_08h49m39s.ibd
```

The filter configuration determines which dataset IDs enter the selection.
`metaspace_annotations.json` controls `annotation_fdr` and whether spatial
annotations are retrieved; it does not define filenames. Retrieved ion images
are converted to molecule-to-spectrum relations and stored in
`datasets/catalog.sqlite`, so no ion-image filename is generated.

For merged output, `output_path` defines the imzML filename and its `.ibd`
companion. `merged_dataset_id` is the catalogue identity of the merged dataset;
it does not rename source dataset directories.

## Detailed instructions

### Authenticate METASPACE

```bash
source assets/scripts/datasets/metaspace_session.sh
```

Source the script in the shell that starts the command or Jupyter process. End
the session with `unset METASPACE_API_KEY`.

### Download all or selected IDs

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py download \
  --workspace-path data/tutorial_workspace \
  --source metaspace \
  --selection data/tutorial_workspace/datasets/selections/metaspace-selection.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json
```

Repeat `--dataset-id DATASET_ID` to restrict a run without changing the
selection. Provider quota failures remain explicit and should not be retried as
successful empty downloads. Authentication failures and responses without a
complete imzML/ibd pair also leave the dataset unmaterialized.
