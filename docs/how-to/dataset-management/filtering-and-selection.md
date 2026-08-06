# Review filters and create a dataset selection

The explorer retains search state while a user reviews records. Exported
filters and manual exclusions create the reproducible input to the query and
materialization operations.

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
provider or local filter rejection.

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

## Export reviewed filters

```python
path = explorer.export_config(
    "data/tutorial_workspace/datasets/filters/metaspace-reviewed.json"
)
```

The exported JSON contains the current provider filters and the union of
configured and manually selected `exclude_dataset_ids`. It does not contain
the displayed `SUMMARY` row or computed pandas formatting.

Execute the reviewed configuration with:

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py query \
  --source metaspace \
  --workspace-path data/tutorial_workspace \
  --filters data/tutorial_workspace/datasets/filters/metaspace-reviewed.json \
  --selection data/tutorial_workspace/datasets/selections/metaspace-selection.json
```

`query_to_selection()` executes the filters, validates normalized source
records, upserts accepted metadata into SQLite, and writes the selection. The
selection records the source, effective filters, accepted datasets, and
annotation threshold. Download consumes this fixed selection and does not
repeat discovery.

Inspect the selection before materialization. In particular, verify dataset
IDs, `annotation_fdr`, file-size coverage, m/z ranges, and whether optional
molecular or spatial statistics were required only for review or are also
needed by later selection logic.

See [Discover external datasets](discovering-datasets.md) for the complete
filter and result interface and
[Selection and materialization internals](../../library-internals/dataset-management/selection-and-materialization.md)
for the persistent handoff.
