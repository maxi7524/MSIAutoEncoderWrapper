# Download and merge with bounded disk use

The combined operation downloads one selected source at a time, stores canonical
metadata and annotations, appends selected spectra, and removes staging files.

## Purpose and available operations

### Disk-use policy

Default operation retains the merged output and catalog while deleting staged
source pairs. `--keep-downloads` preserves all downloaded sources.

Datasets are handled sequentially. Before each provider request, the operation
checks `workspace/datasets/sources/<source>/<dataset_id>/` and reuses a complete
non-empty pair. Otherwise it resumes that dataset in
`workspace/datasets/.staging/<source>/<dataset_id>/`; existing staging files
are not deleted before the provider receives the destination. A reused
canonical source pair is never removed after merge.

### Required selection

The operation consumes the same reviewed selection and annotation options as
standalone download.

## Detailed instructions

### Execute the combined operation

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py download-merge \
  --workspace-path data/tutorial_workspace \
  --source metaspace \
  --selection data/tutorial_workspace/datasets/selections/metaspace-selection.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json \
  --output data/tutorial_workspace/datasets/merged/example/dataset.imzML \
  --merged-dataset-id example \
  --row-width 128 \
  --unannotated-ratio 1.0 \
  --random-seed 0
```

Use repeated `--dataset-id` filters for a subset and `--keep-downloads` when
source pairs are required after merge.

### Handle interrupted runs

The catalog is persistent, but an interrupted merged imzML output should not be
treated as complete. Inspect both output files and the merged mapping before
reusing the result. Source pairs already present in the canonical source or
staging directories are checked separately for every dataset on the next run.
