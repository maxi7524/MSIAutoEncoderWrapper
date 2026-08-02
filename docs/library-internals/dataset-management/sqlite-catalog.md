# SQLite catalog

The SQLite catalog is the canonical local identity, annotation, and provenance
store for external and merged datasets.

## General abstraction

### Table responsibilities

`datasets` owns source identity and metadata. `annotations` owns molecules.
`spectrum_annotations` owns molecule-to-source-spectrum links.
`merged_datasets` and `spectrum_mappings` own merged artifact identity and
reversible indices.

### Transaction boundary

Replacing one dataset's annotations or one merged dataset's mappings occurs as
one database operation so readers do not observe mixed generations.

## Detailed implementation

### Initialize and query

[`DatasetCatalog`](../../../src/msi_autoencoder_wrapper/dataset_management/catalog/sqlite_catalog.py)
initializes schema and lookup indexes on construction. Dataset paths resolve to
source or merged identity after normalization.

### Filter annotations

SQL filters support database name/version, formula, adduct, and maximum FDR.
Spectrum queries join through indexed link tables rather than loading all
annotations and filtering in Python.

### Resolve merged spectra

A merged index maps to source, source dataset, and source spectrum. The SQLite
annotation reader follows that mapping before querying original annotations and
metadata.
