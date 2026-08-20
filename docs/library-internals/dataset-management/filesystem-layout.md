# Dataset-management filesystem layout

Dataset-management files separate large shared source and merged images from
small, cohort-specific configuration, catalogs, and reports. The convention is
owned by
[`DatasetWorkspaceLayout`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/layout.py).

## General abstraction

### Shared image files, cohort-specific bookkeeping

Every canonical imzML/ibd pair — source or composed — lives directly under
`datasets/<id>/`, shared by every cohort that references it. SQLite catalogs,
materialization reports, and composition configuration are cohort-specific
instead, so several cohorts can reference the same downloaded source pair
without duplicating it, while each keeps an independent catalog.

### Two catalog roles, two locations

`catalog_path(cohort_id)` — under `configs/datasets/<cohort_id>/` — is the
**working catalog** that `query` and `download` use for file-materialization
bookkeeping only. `composed_catalog_path(cohort_id)` — under
`datasets/<cohort_id>/`, colocated with the merged imzML — is the
**composed catalog** that `compose` creates, self-contained, with imported
annotations and merged-spectrum provenance. `msi_autoencoder_wrapper`'s
automatic annotation detection looks for the composed form specifically,
because it sits beside the image under the same stem; see
[Attach molecular annotations](../../how-to/data-input-and-preprocessing/annotations.md#supported-sources-and-selection-priority).

### Temporary state

Streaming download requests use no persistent staging directory in the
current design: `download` always materializes directly under
`datasets/<dataset_id>/`.

## Detailed implementation

### Canonical layout

```text
workspace/
├── datasets/
│   ├── <dataset_id>/                    # source pair from download()
│   │   ├── <dataset_id>.imzML
│   │   ├── <dataset_id>.ibd
│   │   ├── annotations.csv              # written by download(), not imported
│   │   └── pixel_intensities.csv        # into any catalog until compose()
│   └── <cohort_id>/                     # compose_cohort() output
│       ├── <cohort_id>.imzML
│       ├── <cohort_id>.ibd
│       ├── <cohort_id>.sqlite           # composed catalog
│       ├── composition.json
│       └── annotation_index.json
└── configs/
    └── datasets/
        └── <cohort_id>/
            ├── <cohort_id>.sqlite       # working catalog
            └── materialization.json     # DatasetWorkspaceLayout.materialization_path();
                                          # download_from_manifest() itself defaults
                                          # its own report next to the selection file
                                          # instead, unless --manifest overrides it
```

Source datasets from every provider share the flat `datasets/<dataset_id>/`
namespace; there is no `datasets/sources/<source>/<dataset_id>/` subdirectory.
A direct `ImzMLMerger.merge()` call (used internally by `compose_cohort()`, or
directly for a custom merge outside the standard cohort pipeline — see
[Compose a cohort dataset](../../how-to/dataset-management/composing-a-cohort.md#custom-spectrum-selection-during-merge))
writes wherever its `output_path` argument points, since it takes that path
directly rather than deriving it from the layout.

Query, filter, and selection JSON files (`filter.json`, `selection.json`) have
no fixed location in the layout; conventionally they are kept alongside the
working catalog under `configs/datasets/<cohort_id>/`, since several CLI
subcommands resolve `<cohort_id>` from the selection file's own parent
directory name — see
[Choose the cohort catalog](../../how-to/dataset-management/downloading-datasets.md#choose-the-cohort-catalog).

### Path resolution

CLI `--filters`/`--selection`/`--config`/`--manifest` arguments are resolved
relative to the invocation directory. Catalog local paths are normalized
before identity comparison.
