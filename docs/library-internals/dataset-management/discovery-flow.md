# Dataset discovery flow

Discovery converts provider search capabilities into reviewed local records and
a versioned selection artifact.

## General abstraction

### Interactive and operational layers

`DatasetExplorer` owns mutable review state. `DatasetSource` owns provider query
semantics. `query_to_selection()` creates the persistent handoff consumed by
materialization.

### No-download boundary

Discovery may retrieve metadata and statistics but does not transfer imzML/ibd
files.

## Detailed implementation

### Explore records

[`DatasetExplorer`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/exploration/dataset_explorer.py)
retains filters, accepted IDs, exclusions, and rejected diagnostics while
delegating query execution to the source. `count_mz_range_coverage()` and
`select_mz_range()` operate on the explorer's current in-memory results;
`filter()`/`search()` run a new provider query and replace them.

> Remark:
> For METASPACE, the reusable catalogue first restricts local free-text matches.
> The adapter then requests current provider records for those IDs, enriches them
> with configuration, size, acquisition, m/z-range, and annotation-count data,
> and optionally calculates molecular or spatial statistics. See
> [METASPACE provider](metaspace-provider.md) for the field-level flow.

### Persist selection

[`query_to_selection()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/query.py)
validates source records, upserts discovered metadata into SQLite, and writes
the effective filters and accepted records. This freezes human review before
network materialization.

[`DatasetExplorer.export_selection()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/exploration/dataset_explorer.py)
is a second, client-side path to the same selection schema: it freezes the
explorer's current accepted records directly, with optional sorting, instead
of repeating the provider query that `query_to_selection()` performs.
