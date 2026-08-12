# Dataset management

`msi-dataset-manager` is an independent, provider-agnostic distribution
(`packages/msi_dataset_manager`, imported as `msi_dataset_manager`) that
discovers, downloads, catalogs, and composes MSI datasets in one canonical
local format. `msi_autoencoder_wrapper` depends on it and consumes its
composed catalog through `SQLiteAnnotationReader`; it does not reimplement
dataset acquisition. The two libraries are documented together here because
they are used together, but they are separate distributions with separate
test suites (`packages/msi_dataset_manager/tests` and the repository
`tests/dataset_sources`).

The supported pipeline is two steps: [download](downloading-datasets.md)
materializes imzML/ibd pairs and their annotation CSVs from a provider without
touching any SQLite catalog; [compose](composing-a-cohort.md) turns a set of
canonical local datasets into one merged, catalog-backed cohort with
molecule-occurrence masks. There is no separate standalone `merge` or
`import-local` command; composition is the only supported path from local
files to a queryable cohort.

```{admonition} METASPACE API access
:class: warning

As of this writing, METASPACE does not allow this project to query or
download datasets through its public API. [Discover datasets](discovering-datasets.md),
[Review and save selections](filtering-and-selection.md), and
[Download datasets](downloading-datasets.md) describe the intended workflow
for when API access is available. Until then, use this module to manage
datasets that are already downloaded:
[Compose a cohort dataset](composing-a-cohort.md) and
[Inspect the catalog](inspecting-the-catalog.md).
```

Every operation is also available from the `msi-datasets` command-line entry
point; see [Use the msi-datasets CLI](command-line-workflow.md) for a
complete command reference.

## Contents

- [Discover datasets](discovering-datasets.md) — query a provider through `DatasetExplorer` or the CLI.
- [Review and save selections](filtering-and-selection.md) — accept, exclude, and export a reproducible selection.
- [Download datasets](downloading-datasets.md) — materialize selected imzML/ibd pairs and annotation CSVs.
- [Retrieve annotations](retrieving-annotations.md) — normalize provider molecular results and spatial links.
- [Compose a cohort dataset](composing-a-cohort.md) — import local annotations, merge canonical local datasets into a cohort, and build molecule occurrence masks.
- [Inspect the catalog](inspecting-the-catalog.md) — query the working and composed SQLite catalogs.
- [Use the msi-datasets CLI](command-line-workflow.md) — every subcommand, its inputs, and its outputs.

```{toctree}
:hidden:

discovering-datasets
filtering-and-selection
downloading-datasets
retrieving-annotations
composing-a-cohort
inspecting-the-catalog
command-line-workflow
```
