# Dataset management

These guides cover the complete external-dataset workflow represented by the
dataset-management tutorials: discovery and review, materialization and
annotation, local import, merging, and catalog inspection.

## Contents

- [Discover datasets](discovering-datasets.md) — query a provider through `DatasetExplorer` or the CLI.
- [Review and save selections](filtering-and-selection.md) — accept, exclude, and export a reproducible selection.
- [Download datasets](downloading-datasets.md) — materialize selected imzML/ibd pairs.
- [Retrieve annotations](retrieving-annotations.md) — normalize provider molecular results and spatial links.
- [Import local datasets](importing-local-datasets.md) — add an imzML pair and paired METASPACE CSV exports.
- [Merge datasets](merging-datasets.md) — select annotated and unannotated spectra and preserve provenance.
- [Download and merge](download-and-merge.md) — process sources sequentially with bounded disk use.
- [Inspect the catalog](inspecting-the-catalog.md) — query canonical SQLite datasets, annotations, and merged mappings.

```{toctree}
:hidden:

discovering-datasets
filtering-and-selection
downloading-datasets
retrieving-annotations
importing-local-datasets
merging-datasets
download-and-merge
inspecting-the-catalog
```
