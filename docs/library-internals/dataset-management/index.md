# Dataset management internals

This section describes provider discovery through canonical storage and merged
artifact provenance.

## Contents

- [Discovery flow](discovery-flow.md) — explorer, source filters, diagnostics, and selection artifacts.
- [Provider boundary](provider-boundary.md) — shared adapter contract and provider-specific responsibilities.
- [Selection and materialization](selection-and-materialization.md) — lifecycle transitions from reviewed IDs to local pairs.
- [Annotation normalization](annotation-normalization.md) — provider and CSV records to canonical molecular and spatial links.
- [SQLite catalog](sqlite-catalog.md) — canonical tables, transactions, filters, and path identity.
- [Merge and provenance](merge-and-provenance.md) — deterministic spectrum selection, output geometry, and reversible mappings.
- [Filesystem layout](filesystem-layout.md) — persistent, staging, selection, source, and merged artifact locations.

```{toctree}
:hidden:

discovery-flow
provider-boundary
selection-and-materialization
annotation-normalization
sqlite-catalog
merge-and-provenance
filesystem-layout
```
