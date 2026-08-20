# Selection and materialization

Materialization consumes a reviewed selection and transitions each dataset from
discovered metadata to validated local files and annotation CSVs. It does not
import annotations into SQLite; that happens only during composition.

## General abstraction

### Selection handoff

The selection contains source, effective filters, reviewed records, and
annotation threshold. Materialization never repeats discovery to silently
change the accepted set.

### Dataset lifecycle

Working-catalog status and local path record discovery and file
materialization. A complete imzML/ibd pair is reusable; a partial pair is not
a valid final state. Annotation CSV completeness is tracked as a plain
filesystem check, not a working-catalog status.

## Detailed implementation

### Download sources

[`download_from_manifest()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/download.py)
validates the selection, optionally restricts IDs, calls the adapter,
validates the imzML pair, retrieves annotations, and writes them as a CSV pair
next to the image — it upserts working-catalog dataset rows for file status
only, never annotation rows. The METASPACE adapter's source-download flow is
detailed in [METASPACE provider](metaspace-provider.md).

### Annotation reuse is a file check

Unlike file materialization, annotation-CSV reuse has no working-catalog
marker: `has_complete_annotation_csv()` simply checks that
`annotations.csv` and `pixel_intensities.csv` both exist and are non-empty.
The `annotation_materializations` table and its `get_annotation_materialization()`/
`record_annotation_materialization()`/`clear_annotation_materializations()`
methods are still defined on `DatasetCatalog` (see
[SQLite catalog](sqlite-catalog.md)) but are not called by the current
`download` or `compose` flow.

### Preserve failures

Provider quota, authentication, missing files, inconsistent FDR, and incomplete
spatial annotations stop the dataset transition. Temporary downloads use partial
paths before replacement where supported.

### Composition is the import step

[`compose_cohort()`](merge-and-provenance.md#compose-a-cohort) reads each
input's CSV pair — writing it into the composed catalog via
`import_local_dataset()` — as part of merging the cohort. A dataset that was
only materialized by `download`, never composed, has no SQLite-queryable
annotations.
