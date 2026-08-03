# Download selected datasets

Materialization downloads imzML/ibd files for accepted selection records and
updates their canonical catalog entries.

## Purpose and available operations

### Required inputs

Download requires a source strategy, selection JSON, workspace dataset
directory, catalog, optional annotation settings, and provider authentication
when required.

### Reuse behavior

A complete non-empty local imzML/ibd pair is reused. An incomplete pair is not
accepted as a materialized dataset.

## Detailed instructions

### Authenticate METASPACE

```bash
source assets/scripts/datasets/metaspace_session.sh
```

Source the script in the shell that starts the command or Jupyter process. End
the session with `unset METASPACE_API_KEY`.

### Download all or selected IDs

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py download \
  --workspace-path data/tutorial_workspace \
  --source metaspace \
  --selection data/tutorial_workspace/datasets/selections/metaspace-selection.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json
```

Repeat `--dataset-id DATASET_ID` to restrict a run without changing the
selection. Provider quota failures remain explicit and should not be retried as
successful empty downloads.
