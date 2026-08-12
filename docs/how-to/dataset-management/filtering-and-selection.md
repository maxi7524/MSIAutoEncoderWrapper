# Review filters and create a dataset selection

```{admonition} METASPACE API access
:class: warning

As of this writing, METASPACE does not allow this project to query datasets
through its public API. This guide documents the intended review and export
workflow for when API access is available.
```

The explorer retains search state while a user reviews records. Exporting that
state creates the reproducible input to the download and materialization
operations.

## Refine the accepted cohort

Use provider filters to repeat discovery when a criterion should be applied to
all candidate records. Use manual exclusion when a known dataset passes the
criteria but is unsuitable for the current experiment.

```python
explorer.exclude(["dataset-a", "dataset-b"])
explorer.include("dataset-b")

accepted = explorer.accepted(summarise=True)
all_records = explorer.accepted(include_excluded=True, summarise=True)
rejected = explorer.rejected()
```

An ID must exist in the current accepted result before it can be excluded.
`include()` removes the ID from the manual exclusion set; it does not bypass a
provider or local filter rejection. `explorer.select_mz_range(min_mz, max_mz)`
(see [Discover external datasets](discovering-datasets.md#explore-mz-coverage-interactively))
excludes datasets the same way and records the applied range in the exported
filters.

The `SUMMARY` row is calculated from the records visible in the returned table.
With `include_excluded=False`, manually excluded records do not contribute to
size totals, m/z intersection, or method unions.

## Distinguish filtering statistics

`annotation_count` is the sum of METASPACE annotation results at
`annotation_fdr` across annotation databases. It is not a unique molecule
count and not an annotated pixel count.

`molecule_count` deduplicates formula-adduct identities within each dataset.
`unique_molecule_count` compares those identities across the current cohort.
Changing biological filters, native filters, dataset IDs, or exclusions can
therefore change which molecules are unique even when the underlying dataset
annotations have not changed.

Spatial statistics use ion images and the same `annotation_fdr`. They are not
used to calculate molecular uniqueness. `unannotated_pixel_count` means an
acquired position with no qualifying METASPACE ion-image signal; it does not
necessarily represent biological background.

## Export a reviewed selection

`export_selection()` freezes the explorer's current accepted records — after
manual exclusions and m/z-range review — into one directory, without
repeating the provider query:

```python
exported = explorer.export_selection(
    "workspace/configs/datasets/kidney",
    sort_by="download_size_bytes",
    ascending=False,
)
exported
# {"filters": .../filter.json, "selection": .../selection.json}
```

This writes two files:

- `filter.json` — the current provider filters and the union of configured and
  manually selected `exclude_dataset_ids`;
- `selection.json` — `schema_version`, `source`, `exported_at`,
  `catalog_retrieved_at`, the filters used, the requested `sort`, and the full
  accepted-dataset records (`dataset_ids` and `datasets`), in the requested
  order.

`sort_by` orders records by any top-level or `metadata`-nested field, for
example `download_size_bytes` to start materialization with the largest
datasets first, or `unique_molecule_count` to prioritize the most
information-dense datasets. Records missing the sort field are appended after
the sorted ones, in their original order. Download and compose operations
consume `selection.json` in the order it stores; they do not re-sort it.

The exported JSON does not contain the displayed `SUMMARY` row or computed
pandas formatting.

## Export filters only

`export_config()` writes only the current filters and exclusions, without
freezing accepted records:

```python
path = explorer.export_config(
    "workspace/configs/datasets/kidney/filter.json"
)
```

Use this when the filters should be re-executed against the provider later,
for example from a script that does not keep the explorer's in-memory state.
Execute the saved configuration with:

```bash
msi-datasets query \
  --source metaspace \
  --workspace-path workspace \
  --filters workspace/configs/datasets/kidney/filter.json \
  --selection workspace/configs/datasets/kidney/selection.json
```

`query_to_selection()` re-executes the filters, validates normalized source
records, upserts accepted metadata into SQLite, and writes a `selection.json`
without a `sort` field or `catalog_retrieved_at` timestamp — datasets appear
in provider response order. Because it repeats the query, its accepted set can
differ from an explorer session's if the provider catalogue changed meanwhile.

Whichever export path is used, inspect the selection before materialization.
In particular, verify dataset IDs, `annotation_fdr`, file-size coverage, m/z
ranges, and whether optional molecular or spatial statistics were required
only for review or are also needed by later selection logic.

See [Discover external datasets](discovering-datasets.md) for the complete
filter and result interface and
[Selection and materialization internals](../../library-internals/dataset-management/selection-and-materialization.md)
for the persistent handoff.
