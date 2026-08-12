# Dataset management internals

`msi_dataset_manager` (`packages/msi_dataset_manager`) is an independent
distribution: provider discovery through canonical storage and merged
artifact provenance are implemented there, with no dependency on
`msi_autoencoder_wrapper`. The wrapper depends on it and reads its SQLite
catalog through `SQLiteAnnotationReader`
(`src/msi_autoencoder_wrapper/annotations/strategies/sqlite_annotation_reader.py`);
it does not implement any of the flows described in this section. Links below
point into `packages/msi_dataset_manager/src/msi_dataset_manager`.

## Contents

- [Discovery flow](discovery-flow.md) — explorer, source filters, diagnostics, and selection artifacts.
- [Provider boundary](provider-boundary.md) — shared adapter contract and provider-specific responsibilities.
- [METASPACE provider](metaspace-provider.md) — METASPACE API capabilities, wrapper enrichment, statistics, progress, and persistence flow.
- [Selection and materialization](selection-and-materialization.md) — lifecycle transitions from reviewed IDs to local pairs.
- [Annotation normalization](annotation-normalization.md) — provider and CSV records to canonical molecular and spatial links.
- [SQLite catalog](sqlite-catalog.md) — canonical tables, transactions, filters, and path identity.
- [Merge and provenance](merge-and-provenance.md) — deterministic spectrum selection, output geometry, and reversible mappings.
- [Filesystem layout](filesystem-layout.md) — persistent, staging, cohort-config, and merged/composed artifact locations.

```{toctree}
:hidden:

discovery-flow
provider-boundary
metaspace-provider
selection-and-materialization
annotation-normalization
sqlite-catalog
merge-and-provenance
filesystem-layout
```
