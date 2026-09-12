# Use local metabolite databases

This guide describes how to store external metabolite exports locally and use
them to build dataset-specific candidate-ion catalogs.

## Scope

The complete source databases are stored in
`assets/local/metabolite_databases/`. Dataset-specific filters are applied when
`candidates.sqlite` is built; the raw source exports are not modified.

The internal separation between raw sources, candidate catalogs, and synthetic
sampling is described in [Metabolite candidate catalogs](../../library-internals/dataset-management/metabolite-candidate-catalog.md).

## Materialize provider exports

Run the provider command from the repository root:

```bash
uv run --project packages/msi_dataset_manager \
  msi-datasets candidate-sources
```

The default output directory is `assets/local/metabolite_databases/`. The
command reuses existing source files and writes `sources_manifest.json` with
provider names, versions, URLs, sizes, and local paths.

To refresh selected providers, use `--provider` and `--refresh-cache`:

```bash
uv run --project packages/msi_dataset_manager \
  msi-datasets candidate-sources \
  --provider lipidmaps \
  --provider hmdb \
  --refresh-cache
```

The supported provider identifiers are `lipidmaps`, `hmdb`, and `chebi`.

## Register a downloaded HMDB export

The HMDB source export is retained as both the original ZIP and the extracted
XML:

```text
assets/local/metabolite_databases/hmdb_5-0/
├── hmdb_metabolites.zip
└── hmdb_metabolites.xml
```

The provider reads the XML snapshot directly. This avoids unpacking the 6.1 GB
XML during every candidate-catalog build.

## Build a dataset-specific candidate catalog

Use `build_candidate_catalog()` after normalized dataset metadata has been
materialized:

```python
from msi_dataset_manager.annotations.candidates import (
    HMDBCandidateProvider,
    LIPIDMAPSCandidateProvider,
    ChEBICandidateProvider,
    build_candidate_catalog,
)

catalog_path = build_candidate_catalog(
    workspace_path="workspace",
    dataset_id="liver",
    providers=(
        LIPIDMAPSCandidateProvider(version="retrieved-2026-09-12"),
        HMDBCandidateProvider(version="5.0"),
        ChEBICandidateProvider(version="255"),
    ),
    adducts=("+H", "+Na", "-H"),
)
```

When `source_cache_dir` is omitted, providers look in
`assets/local/metabolite_databases/`. The resulting filtered catalog is stored
under the dataset's `database_annotations/candidates.sqlite` path. Dataset
metadata supplies polarity and `m/z` limits when they are available.

Use `refresh_cache=True` to rebuild the dataset-specific SQLite catalog. Use
`refresh_sources=True` only when the raw provider exports must also be
replaced.

## Verify the result

```python
from msi_dataset_manager.annotations.candidates import CandidateCatalogReader

reader = CandidateCatalogReader(catalog_path)
print(reader.get_manifest())
ions = reader.get_candidate_ions(
    {"polarity": "Positive", "mz_min": 100, "mz_max": 1000}
)
print(len(ions))
```

The manifest records the source versions and effective polarity, `m/z`, and
adduct filters. The candidate reader supports additional provider, formula,
adduct, polarity, and `m/z` filters without re-reading external databases.
