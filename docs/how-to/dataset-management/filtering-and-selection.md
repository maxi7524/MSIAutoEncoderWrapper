# Review filters and create a dataset selection

The explorer exports reviewed filters and manual exclusions. The query operation
then creates the selection containing accepted dataset records for download.

## Purpose and available operations

### Review state

Explorer results can be excluded and included by dataset ID. Accepted results
omit excluded records by default; diagnostic views retain rejection reasons.

### Two persistent artifacts

`DatasetExplorer.export_config()` writes query filters. `query_to_selection()`
executes those filters, stores accepted metadata, and writes the selection.
Download consumes the selection and does not repeat discovery.

## Detailed instructions

### Modify the accepted set

```python
explorer.exclude(["dataset-a", "dataset-b"])
explorer.include("dataset-b")

accepted = explorer.accepted()
all_records = explorer.accepted(include_excluded=True)
rejected = explorer.rejected()
```

IDs must exist in the current result set. Including an ID removes its explicit
exclusion; it does not bypass provider validation failures.

### Export reviewed filters

```python
path = explorer.export_config(
    "data/tutorial_workspace/datasets/filters/metaspace-reviewed.json"
)
```

The exported mapping includes current filters and `exclude_dataset_ids`. Pass it
to the query CLI:

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py query \
  --source metaspace \
  --workspace-path data/tutorial_workspace \
  --filters data/tutorial_workspace/datasets/filters/metaspace-reviewed.json \
  --selection data/tutorial_workspace/datasets/selections/metaspace-selection.json
```

Review the generated selection's source, filters, accepted records, and
annotation FDR before download.
