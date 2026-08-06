# Merge and provenance

Merge combines selected spectra while preserving enough identity to recover
their source dataset and annotations.

## General abstraction

### Selection ownership

Spectrum-selection logic is separate from imzML writing. It combines all
annotated IDs with deterministic optional sampling of unannotated candidates.

### Geometry and provenance

Merged rectangular coordinates are an output layout, not original tissue
geometry. Source identity is retained in user parameters and SQLite mappings.

## Detailed implementation

### Select spectra

[`select_merge_spectrum_ids()`](../../../src/msi_autoencoder_wrapper/dataset_management/operations/spectrum_selection.py)
uses a seed namespace containing source identity so sampling is deterministic
per dataset rather than globally order-dependent.

### Write output

[`ImzMLMerger`](../../../src/msi_autoencoder_wrapper/dataset_management/operations/merge.py)
appends spectra, assigns row-width coordinates, registers the merged artifact,
and replaces mappings after successful writing.

### Stream download and merge

The combined operation stages one source, normalizes annotations, appends its
selection, and cleans staging. Partial merged files are removed on failure;
canonical source annotations remain in SQLite. For METASPACE, source download
and annotation retrieval within this stream are described in
[METASPACE provider](metaspace-provider.md).
