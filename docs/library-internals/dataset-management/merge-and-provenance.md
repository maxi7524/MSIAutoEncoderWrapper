# Merge and provenance

Merge combines selected spectra while preserving enough identity to recover
their source dataset and annotations. There is no standalone `merge` or
`download-merge` CLI command; [`compose_cohort()`](#compose-a-cohort) is the
supported entry point, and calls the lower-level pieces below internally.

## General abstraction

### Selection ownership

Spectrum-selection logic is separate from imzML writing. It combines all
annotated IDs with deterministic optional sampling of unannotated candidates.

### Geometry and provenance

Merged rectangular coordinates are an output layout, not original tissue
geometry. Source identity is retained in user parameters and SQLite mappings.

## Detailed implementation

### Select spectra

[`select_merge_spectrum_ids()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/spectrum_selection.py)
uses a seed namespace containing source identity so sampling is deterministic
per dataset rather than globally order-dependent.

### Write output

[`ImzMLMerger`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/merge.py)
appends spectra, assigns row-width coordinates, registers the merged artifact,
and replaces mappings after successful writing. It is importable only from the
`operations.merge` submodule, not the top-level `operations` package, marking
it as an internal building block rather than a primary public entry point;
see [Compose a cohort dataset](../../how-to/dataset-management/composing-a-cohort.md#custom-spectrum-selection-during-merge)
for direct use.

### Compose a cohort

[`compose_cohort()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/composition.py)
is the one supported path from canonical local datasets — each with a
materialized imzML/ibd pair and, optionally, an `annotations.csv` /
`pixel_intensities.csv` pair — to a merged, catalog-backed cohort. Per input,
it:

```text
validate the local imzML/ibd pair
  -> missing pair: record in missing_dataset_ids, skip
has_complete_annotation_csv()?
  -> yes: import_local_dataset() into the composed catalog
  -> no: upsert_dataset(status="materialized_without_annotations")
ImzMLMergeInput(source, dataset_id, imzml_path)
```

then merges every collected input with `ImzMLMerger`, using `max_fdr` to
decide which spectra count as annotated for `unannotated_ratio`/
`unannotated_amount` sampling, and finally calls
[`build_cohort_annotation_index()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/cohort_annotations.py)
with the same `max_fdr`/`minimum_dataset_occurrence` against the composed
catalog to derive cohort-wide molecule-occurrence masks — a molecule observed
in only one dataset is retained and flagged, not removed. It writes
`composition.json` and `annotation_index.json` next to the merged output.

The composed catalog it creates,
[`DatasetWorkspaceLayout.composed_catalog_path()`](filesystem-layout.md), is a
different file from the working catalog
([`catalog_path()`](filesystem-layout.md)) that `query`/`download` use; it is
self-contained and colocated with the merged image specifically so
`msi_autoencoder_wrapper`'s same-stem sibling-catalog auto-detection finds it
without configuration.
