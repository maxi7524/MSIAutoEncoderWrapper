# Download selected datasets

```{admonition} METASPACE API access
:class: warning

As of this writing, METASPACE does not allow this project to download
datasets through its public API. This guide documents the intended
materialization workflow for when API access is available.
```

Materialization downloads imzML/ibd files for accepted selection records and
writes their annotations as CSV files beside them. It does **not** import
annotations into any SQLite catalog — that happens only when the dataset is
later composed into a cohort; see [Compose a cohort dataset](composing-a-cohort.md).
The METASPACE source-download flow is described in
[METASPACE provider internals](../../library-internals/dataset-management/metaspace-provider.md).

## Purpose and available operations

### Required inputs

Download requires a source strategy, a selection JSON (from
[Review and save selections](filtering-and-selection.md)), a workspace path,
optional annotation retrieval options, and provider authentication when
required.

### Two-phase materialization

`materialize_selection()` processes the whole selection in two passes: it
downloads or reuses every dataset's imzML/ibd pair first, then retrieves or
reuses every materialized dataset's annotation CSVs. A provider quota error
during the first pass stops further file downloads but does not skip the
annotation pass for datasets whose pair already completed before the error —
the operation searches the remaining selection records once more for
already-complete local pairs before starting annotation retrieval.

### File reuse behavior

Before requesting source files, the operation checks
`workspace/datasets/<dataset_id>/`. A complete non-empty local imzML/ibd pair
bypasses the provider download operation. If the pair is incomplete, only that
dataset is passed to the provider; the official METASPACE client can skip an
existing member and transfer its missing companion. The wrapper validates the
complete pair afterward.

### Annotation output and reuse behavior

Annotation retrieval writes two canonical CSV files beside the imzML pair:
`annotations.csv` (one row per molecular annotation) and
`pixel_intensities.csv` (per-spectrum intensity, one column per acquired
position). Reuse is a direct file check —
`has_complete_annotation_csv()` — not a catalog lookup: if both files already
exist and are non-empty for a dataset, retrieval is skipped for it and the run
reuses them as-is, regardless of what retrieval options originally produced
them. Delete the CSV pair to force re-retrieval with different
`--annotation-options`.

Ion images used for spatial annotations are not cached as individual files.
Retrieval always requests the fully qualifying set of images for a dataset
whose CSV pair is not already complete.

### Output names and layout

The source filename comes from the stable `dataset_id` stored in the
selection, not from the dataset display name. For a workspace at `workspace`,
dataset `2026-07-27_08h49m39s` is stored as:

```text
workspace/datasets/
└── 2026-07-27_08h49m39s/
    ├── 2026-07-27_08h49m39s.imzML
    ├── 2026-07-27_08h49m39s.ibd
    ├── annotations.csv
    └── pixel_intensities.csv
```

Source datasets from every provider share this flat `datasets/<dataset_id>/`
namespace; there is no per-provider subdirectory. See
[Dataset-management filesystem layout](../../library-internals/dataset-management/filesystem-layout.md)
for the complete layout, including where the working SQLite catalog and
materialization report are written.

The filter configuration determines which dataset IDs enter the selection.
`--annotation-options` controls `annotation_fdr` and whether spatial
annotations are retrieved; it does not affect filenames.

## Detailed instructions

### Authenticate METASPACE

```bash
source assets/scripts/datasets/metaspace_session.sh
```

Source the script in the shell that starts the command or Jupyter process. End
the session with `unset METASPACE_API_KEY`.

### Download a selection

```bash
msi-datasets download \
  --workspace-path workspace \
  --source metaspace \
  --selection workspace/configs/datasets/kidney/selection.json \
  --annotation-options workspace/configs/datasets/kidney/annotation_options.json
```

Repeat `--dataset-id DATASET_ID` to restrict a run without changing the
selection. Provider quota failures remain explicit and should not be retried
as successful empty downloads. Authentication failures and responses without a
complete imzML/ibd pair also leave the dataset unmaterialized.

`--annotation-options` is a JSON object, for example:

```json
{"annotation_fdr": 0.1, "include_spatial": true}
```

If the selection's stored `filters.annotation_fdr` and the supplied
`annotation_fdr` disagree, the command raises a validation error rather than
silently choosing one. Omit the option to reuse the selection's own
`annotation_fdr`.

### Rotate provider profiles

`--profiles PROFILES.csv` supplies more than one API key so a quota error on
one account continues on the next instead of stopping the run:

```text
key,comment
first-secret-key,primary account
second-secret-key,backup account
```

Only the `key` column is required; other columns are free-form operator
metadata and are not read by the CLI. Only a quota error (`DownloadLimitError`)
advances to the next profile; authentication, validation, and network errors
remain visible immediately. Keys are never written to the manifest or logs.

### Choose the cohort catalog

`query` and `download` write to a **working catalog** at
`workspace/configs/datasets/<cohort_id>/<cohort_id>.sqlite`, which tracks file
materialization status only — it holds no imported annotations.
`--cohort-id` sets `<cohort_id>` explicitly; otherwise it defaults to the
selection file's parent directory name (so keeping each cohort's
`selection.json` in its own `workspace/configs/datasets/<cohort_id>/`
directory, as
[`export_selection()`](filtering-and-selection.md#export-a-reviewed-selection)
does, selects the matching working catalog automatically), or to `"default"`
if that cannot be determined. This is a different file from the
**composed catalog** that [`compose`](composing-a-cohort.md) writes under
`workspace/datasets/<cohort_id>/<cohort_id>.sqlite` — the one that actually
holds imported annotations and that `msi_autoencoder_wrapper` reads.

### Inspect the materialization report

Unless `--manifest PATH` is given, a report is written next to the selection
file as `<selection>.parent/materialization.json`. It records, per dataset,
whether files and annotation CSVs were downloaded or reused, any failures, and
how many profiles were exhausted. It does not contain provider secrets.
