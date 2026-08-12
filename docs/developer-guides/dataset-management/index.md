# Dataset management development

`msi_dataset_manager` is developed as a separate, independent distribution
under `packages/msi_dataset_manager` (its own `pyproject.toml`, source under
`packages/msi_dataset_manager/src/msi_dataset_manager`, and tests under
`packages/msi_dataset_manager/tests`). It has no dependency on
`msi_autoencoder_wrapper` and must not gain one; the wrapper depends on it
instead, through `SQLiteAnnotationReader`. Changes to dataset management are
made in that package, then validated against the wrapper's own
`tests/dataset_sources` integration tests, which exercise the installed
`msi_dataset_manager` the way the wrapper actually uses it.

These guides extend external provider discovery and canonical local dataset
state.

## Contents

- [Add a dataset provider](adding-a-dataset-provider.md) — implement and register another external database adapter.
- [Add provider filters](adding-provider-filters.md) — expose native and derived filter semantics with diagnostics.
- [Extend annotation normalization](extending-annotation-normalization.md) — map provider annotations into canonical molecules and spectrum links.
- [Change the catalog](changing-the-catalog.md) — migrate canonical SQLite state and preserve reader behavior.

```{toctree}
:hidden:

adding-a-dataset-provider
adding-provider-filters
extending-annotation-normalization
changing-the-catalog
```
