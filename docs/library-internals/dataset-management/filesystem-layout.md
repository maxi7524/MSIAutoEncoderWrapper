# Dataset-management filesystem layout

Dataset-management files separate reviewed selections, persistent source pairs,
merged outputs, and temporary staging.

## General abstraction

### Persistent state

The workspace dataset root owns `catalog.sqlite`, selections, source material,
and merged artifacts. SQLite remains the metadata authority even when staged
source files are deleted.

### Temporary state

Streaming operations use `.staging`. Temporary files are not valid user-facing
dataset paths.

## Detailed implementation

### Canonical layout

```text
datasets/
├── catalog.sqlite
├── selections/
├── sources/<source>/<dataset_id>/
│   ├── <dataset_id>.imzML
│   └── <dataset_id>.ibd
├── merged/<merged_dataset_id>/
│   ├── dataset.imzML
│   └── dataset.ibd
└── .staging/<source>/<dataset_id>/
```

### Path resolution

Selection/config paths are resolved relative to their declaring file where the
CLI specifies that behavior. Workspace defaults resolve from project root.
Catalog local paths are normalized before identity comparison.
