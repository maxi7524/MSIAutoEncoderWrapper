# Dataset management development

These guides extend external provider discovery and canonical local dataset
state.

## Contents

- [Add a dataset provider](adding-a-dataset-provider.md) — implement and register another external database adapter.
- [Add provider filters](adding-provider-filters.md) — expose native and derived filter semantics with diagnostics.
- [Extend annotation normalization](extending-annotation-normalization.md) — map provider annotations into canonical molecules and spectrum links.
- [Change the catalog](changing-the-catalog.md) — migrate canonical SQLite state and preserve reader behavior.

The earlier [combined provider guide](../adding-a-dataset-source.md) remains
available during review.

```{toctree}
:hidden:

adding-a-dataset-provider
adding-provider-filters
extending-annotation-normalization
changing-the-catalog
```
