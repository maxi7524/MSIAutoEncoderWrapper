# SQLite catalog

The SQLite catalog is the canonical local identity, annotation, and provenance
store for external and merged datasets. `DatasetCatalog` is one schema used
at two distinct paths per cohort, with different roles — see
[Filesystem layout](filesystem-layout.md):

- the **working catalog** (`DatasetWorkspaceLayout.catalog_path()`), written by
  `query`/`download`, holds only source-dataset identity and file
  materialization status;
- the **composed catalog** (`DatasetWorkspaceLayout.composed_catalog_path()`),
  written by `compose`, is self-contained and additionally holds imported
  annotations and merged-spectrum mappings.

## General abstraction

### Table responsibilities

`datasets` owns source identity and metadata. `annotations` owns molecules.
`spectrum_annotations` owns molecule-to-source-spectrum links.
`merged_datasets` and `spectrum_mappings` own merged artifact identity and
reversible indices. `annotation_materializations` is defined in the schema but
currently has no caller in the `download`/`compose` flow: annotation-CSV reuse
during download is a plain filesystem check
(`has_complete_annotation_csv()`), not a catalog marker — see
[Selection and materialization](selection-and-materialization.md).

### Transaction boundary

Replacing one dataset's annotations or one merged dataset's mappings occurs as
one database operation so readers do not observe mixed generations.

## Detailed implementation

### Initialize and query

[`DatasetCatalog`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/catalog/sqlite_catalog.py)
initializes schema and lookup indexes on construction. Dataset paths resolve to
source or merged identity after normalization. The METASPACE adapter's
selection- and materialization-time writes to the working catalog, and
composition's writes to the composed catalog, are described in
[METASPACE provider](metaspace-provider.md) and
[Merge and provenance](merge-and-provenance.md).

### Filter annotations

SQL filters support database name/version, formula, adduct, and maximum FDR.
Spectrum queries join through indexed link tables rather than loading all
annotations and filtering in Python. These queries only return rows against a
composed catalog; a working catalog has no `annotations` rows to match.

### Resolve merged spectra

A merged index maps to source, source dataset, and source spectrum. The SQLite
annotation reader follows that mapping before querying original annotations and
metadata.
