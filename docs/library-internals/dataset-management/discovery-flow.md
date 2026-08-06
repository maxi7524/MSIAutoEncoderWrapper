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

[`DatasetExplorer`](../../../src/msi_autoencoder_wrapper/dataset_management/exploration/dataset_explorer.py)
retains filters, accepted IDs, exclusions, and rejected diagnostics while
delegating query execution to the source.

> Remark:
> For METASPACE, the reusable catalogue first restricts local free-text matches.
> The adapter then requests current provider records for those IDs, enriches them
> with configuration, size, acquisition, m/z-range, and annotation-count data,
> and optionally calculates molecular or spatial statistics. See
> [METASPACE provider](metaspace-provider.md) for the field-level flow.

### Persist selection

[`query_to_selection()`](../../../src/msi_autoencoder_wrapper/dataset_management/operations/query.py)
validates source records, upserts discovered metadata into SQLite, and writes
the effective filters and accepted records. This freezes human review before
network materialization.
