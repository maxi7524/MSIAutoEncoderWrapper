<!-- #DONE: Revised -->

# METASPACE provider

The METASPACE provider translates the official Python client and additional
GraphQL fields into the dataset-source contract used by discovery,
materialization, annotation normalization, and cohort review.

## General abstraction

### Provider boundary

[`MetaspaceDatasetSource`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace.py)
is the adapter entry point. Dataset-management operations depend on the
provider-independent
[`DatasetSource`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/base.py)
contract and do not call METASPACE directly.

The adapter owns provider authentication, client argument names, GraphQL
payloads, pagination, annotation database iteration, ion-image interpretation,
download-link validation, and translation of provider exceptions. 

> Remark:
> Credentials are resolved by [`metaspace_authentication.py`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace_authentication.py) and are not stored in records, cache files, filters, or selections.

### Discovery and materialization boundary

Discovery retrieves metadata, annotation identities, and optionally ion
images. It does not retrieve source imzML or ibd files. Materialization uses a
reviewed selection and calls `download_dataset()` for the source pair.

METASPACE metadata requests and ion-image requests are therefore network work,
but only source-pair transfer changes the local dataset from discovered to
materialized. This distinction is preserved in catalogue status and local path
state.

### Metadata representations

One normalized source record has three top-level fields:

```python
{
    "dataset_id": "2026-04-22_21h03m00s",
    "name": "provider dataset name",
    "metadata": {...},
}
```

`metadata` retains the original METASPACE metadata sections and adds stable
wrapper keys. Raw provider fields remain under `provider_metadata`. 

Consumersn use normalized keys such as `condition`, `organism_part`, `analyzer`,
`total_size_bytes`, `mz_min`, and `molecule_count` instead of parsing private
client objects.

### Cost classes

Catalogue search, current dataset configuration, file sizes, acquisition
geometry, diagnostics, and annotation counts use JSON API responses. Molecular
statistics retrieve annotation identities but not image arrays.

Spatial statistics are more expensive because they retrieve first-isotope ion
images for all qualifying annotations. Source download is independent and
retrieves the complete imzML/ibd pair only during materialization.

### Failure boundaries

Provider authentication and GraphQL failures stop the current discovery.
Missing optional metadata produces missing normalized fields. Invalid spatial
relationships, incomplete annotation images, unsupported download contents,
and incomplete source pairs raise errors because continuing would create
incorrect counts or materialized state.

## Detailed implementation

### What METASPACE provides

#### Python client entry points

The project pins `metaspace2020==2.0.9`. The official
[`SMInstance`](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html#metaspace.sm_annotation_utils.SMInstance)
constructs authenticated API access.
[`SMInstance.datasets()`](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html#metaspace.sm_annotation_utils.SMInstance.datasets)
searches datasets, and `SMInstance.dataset(id=...)` resolves one
[`SMDataset`](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html#metaspace.sm_annotation_utils.SMDataset).

The exact public-wrapper-to-client argument mapping is declared in
[`metaspace_parameters.py`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace_parameters.py).
The distinction matters because Python arguments such as `ionisation_source`
and `analyzer_type` differ from the underlying GraphQL fields
`ionisationSource` and `analyzerType`.

#### Dataset catalogue

`SMInstance.datasets(status="FINISHED")` supplies the accessible catalogue.
`MetaspaceDatasetSource.__init__()` loads this catalogue or the versioned
`available-datasets.json` cache. The cache contains normalized records, not
client objects or credentials.

The cache supplies `get_available_values()` and preselection for local
free-text filters. It is not treated as authoritative for current annotation
counts, processing configuration, sizes, or diagnostics.

#### Dataset metadata

`SMDataset.metadata`, `SMDataset.polarity`, `SMDataset.status`,
`SMDataset.image_size`, and `SMDataset.database_details` provide the documented
dataset surface. Private `_info` is retained as `provider_metadata` because the
pinned client exposes submitter, group, projects, analyzer, ionization source,
acquisition geometry, and other dataset-list fields there.

`_dataset_record()` in
[`metaspace.py`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace.py)
combines these sources. Sample fields fall back to the structured
`Sample_Information` and `MS_Analysis` metadata sections when the dataset-list
field is absent.

> Remark:
> We created this part based on the official 
> [dataset metadata example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-dataset-metadata.html)
> shows the client-level metadata interface.

#### Configuration, geometry, sizes, and diagnostics

The METASPACE dataset GraphQL type exposes `configJson`,
`acquisitionGeometry`, `sizeHash`, and `diagnostics`. These fields are also used
by the METASPACE web interface. Their upstream definitions are in the
[`Dataset` GraphQL schema](https://github.com/metaspace2020/metaspace/blob/master/metaspace/graphql/schemas/dataset.graphql).

[`attach_api_metadata()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace_metadata.py)
requests these fields in ID batches. It derives:

- `mz_tolerance_ppm` from `configJson.image_generation.ppm`;
- `analysis_version` and `analysis_method` from
  `configJson.analysis_version`;
- `pixel_count` from `acquisitionGeometry.pixel_count`, falling back to the
  diagnostic spectrum count;
- `imzml_size_bytes` and `ibd_size_bytes` from `sizeHash`;
- `total_size_bytes` and `download_size_bytes` as their sum;
- `mz_min` and `mz_max` from the `IMZML_METADATA` diagnostic.

GraphQL returns diagnostic `data` as a JSON string in the pinned API. The
normalizer deserializes it before reading `min_mz`, `max_mz`, and `n_spectra`.
The range is generated from the imzML reader by METASPACE's
[`extract_dataset_diagnostics()`](https://github.com/metaspace2020/metaspace/blob/master/metaspace/engine/sm/engine/annotation/diagnostics.py),
so it describes observed source spectra rather than the range of annotated
molecules.

File sizes are read from `sizeHash`. Discovery does not request signed download
links or perform HTTP `HEAD` requests to calculate them.

#### Annotation counts

`_attach_annotation_counts()` submits a paginated GraphQL query for selected
dataset IDs and the requested `annotation_fdr`. Counts are retained per
database and summed into `annotation_count`. Optical-image availability is
normalized from the same response.

An annotation may occur in more than one database, so `annotation_count` is a
result count rather than a distinct chemical identity count.

#### Molecular annotations

The official client documents
[`SMDataset.annotations()`](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html#metaspace.sm_annotation_utils.SMDataset.annotations)
and
[`SMDataset.results()`](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html#metaspace.sm_annotation_utils.SMDataset.results).

`_attach_molecule_statistics()` uses the client's batched GraphQL annotation
access to retrieve dataset ID, sum formula, and adduct at `annotation_fdr`.
`get_annotations()` uses `SMDataset.results()` per configured molecular
database when complete molecular rows are required for materialization.

#### Ion images

METASPACE provides
[`SMDataset.isotope_images()`](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html#metaspace.sm_annotation_utils.SMDataset.isotope_images)
and grouped annotation-image retrieval. The official
[isotopic-image example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-isotopic-images.html)
describes the provider representation.

The wrapper requests only the first isotope, disables provider intensity
scaling, and interprets finite non-zero values as annotated positions. The
official client's nested progress bar is suppressed because the wrapper owns
the persistent overall and transient current-operation progress views.

`SMDataset.all_annotation_images()` first retrieves annotation records and
their image metadata, then uses the official client's worker pool to retrieve
the image URLs. With `only_first_isotope=True`, this normally produces one
image request per qualifying annotation rather than one archive request for
the dataset. Image retrieval and source imzML/ibd download are independent
provider operations; an image-request failure does not indicate that the
source-file download quota was reached.

The adapter keeps returned image arrays only for the current normalization
operation. It converts annotated positions to catalogue relations and does not
maintain a file-backed image cache or partial-image manifest. Consequently,
the provider client cannot skip images retrieved by an interrupted earlier
operation.

#### Source downloads

`SMDataset.download_links()` returns provider-authorized imzML and ibd links.
`download_dataset()` validates that the response describes the supported
source pair, then delegates signed-link handling, concurrent transfer, and file
naming to the official `SMDataset.download_to_dir()` method. The adapter checks
the final pair before returning. Existing non-empty pairs are reused.

The materialization layer supplies `datasets/<dataset_id>/` (shared across
every provider — see [Filesystem layout](filesystem-layout.md)) as the
destination and the stable dataset ID as `base_name`. The resulting filenames
are therefore `<dataset_id>.imzML` and `<dataset_id>.ibd`. Annotation options
do not participate in source-file naming.

Both materialization modes inspect the canonical workspace pair before calling
the provider adapter. Selection records are processed sequentially, so one
missing dataset produces one provider download operation. Combined
download-and-merge also preserves a partially populated per-dataset staging
directory, allowing the official client to request only the missing member of
the source pair.

Authentication, quota, missing-link, unsupported-file, and incomplete-pair
conditions are translated into project exceptions before catalogue state is
updated.

### What the wrapper introduces

#### Explicit filtering

[`filter_schema()` and `split_filters()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace_parameters.py)
define every accepted public filter and exact client argument. Unknown keys
raise `ValueError`. Defaults such as `status="FINISHED"` and
`annotation_fdr=0.1` are inserted at this boundary. The convenience pair
`mz_min`/`mz_max` is folded into `mz_range: {"min", "max", "mode": "covers"}`
at this same boundary; supplying both forms together raises `ValueError`.

Free-text groups are declared as canonical PascalCase values with explicit raw
variants. The same table drives available-value grouping and filtering, so the
displayed choice and actual match semantics cannot diverge.

Before a provider query, cached records are restricted by dataset IDs and
exact cached native fields, then by local biological groups. The matched IDs
are passed as `idMask`. Local filtering is repeated on the current provider
records to protect against stale cache contents.

`_apply_early_filters()` then applies `exclude_dataset_ids` and the `mz_range`
coverage requirement together, immediately after metadata and m/z ranges are
attached and before annotation counts, molecular statistics, or spatial
statistics are requested — datasets rejected by either condition never reach
those more expensive stages.

#### Normalized acquisition and biological fields

`_dataset_record()` selects provider-list fields first and structured sample
metadata second. `DatasetExplorer._summary_row()` presents condition separately
from disease metadata and exposes instrument, ionization source, analyzer type,
and resolving power as distinct columns.

No disease is inferred from `condition`. This preserves values such as
`Wildtype`, `Control`, and sample-preparation states without assigning a false
disease interpretation.

#### Distinct and cohort-unique molecules

For each dataset, `_attach_molecule_statistics()` creates a set of
`(sumFormula, adduct)` identities. `molecule_count` is the size of this set.

The function counts each identity's dataset occurrence across the current
cohort. An identity contributes to `unique_molecule_count` only when its
occurrence count is one. `unique_molecules` stores formatted formula-adduct
labels. Uniqueness is therefore relative to the filtered cohort and must be
recalculated when cohort membership changes.

#### Spatial annotation statistics

`_attach_spatial_annotation_statistics()` computes a boolean union across
qualifying first-isotope images. Image shapes must agree. The number of true
positions becomes `annotated_pixel_count`.

When acquisition geometry is available, `unannotated_pixel_count` is
`pixel_count - annotated_pixel_count`, and `annotated_pixel_fraction` divides
the annotated count by acquired pixels. Rectangular image dimensions are not
used as a substitute for acquired-pixel count because an image rectangle can
contain unmeasured positions.

The wrapper also records the number of image annotations, the number of
contributing databases, the applied FDR, and whether the result is complete or
missing acquisition geometry.

#### Progress reporting

`MetaspaceDatasetSource.filter()` plans catalogue, metadata, annotation-count,
optional molecular, and per-dataset spatial operations. One persistent bar
reports completed operations. One transient bar identifies only the current
operation. When spatial retrieval is enabled, the persistent bar reports the
total selected datasets and annotation images.

#### Cohort summaries

[`summarise_records()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/sources/strategies/metaspace_metadata.py)
computes:

- dataset count;
- sum of known source sizes and a completeness flag;
- intersection `[max(mz_min), min(mz_max)]` when all ranges exist and overlap;
- union of analysis methods;
- union of ionization sources and analyzer types.

`DatasetExplorer.results(summarise=True)` converts this mapping into a final
`SUMMARY` row. The row is a presentation record and is not included in source
records, exported filters, selections, or SQLite dataset rows.

### Discovery data flow

The interactive flow is:

```text
DatasetExplorer.filter(filters)
  -> MetaspaceDatasetSource.filter(filters)
     -> split_filters()
     -> restrict cached catalogue and construct idMask
     -> SMInstance.datasets()
     -> _dataset_record() for each SMDataset
     -> attach_api_metadata()
     -> apply current free-text filters
     -> _apply_early_filters(): exclude_dataset_ids and mz_range coverage
     -> attach annotation counts
     -> optionally retrieve spatial ion images
     -> optionally retrieve molecular identities
     -> apply quantitative constraints
     -> store accepted and rejected records
  -> validate_source_record() for each accepted record
  -> DatasetExplorer.results()
     -> excludes IDs added through explorer.exclude(), independently of
        exclude_dataset_ids already applied above
  -> optionally append SUMMARY
```

`exclude_dataset_ids` in `filters` and IDs passed to `explorer.exclude()` are
two independent exclusion paths. The former reaches the METASPACE adapter
(non-METASPACE sources have it stripped before the provider call and rely on
the explorer's own post-filtering instead) and is diagnosed by
`_apply_early_filters()`; the latter never leaves the explorer and only
affects `DatasetExplorer.results()`/`accepted()`, which is why
`include_excluded=True` can still surface it.

The cache reduces the provider candidate set but does not bypass current API
metadata or annotation queries.

### Selection persistence

[`query_to_selection()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/query.py)
executes the source query, validates accepted records, and upserts dataset
identity and metadata into
[`DatasetCatalog`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/catalog/sqlite_catalog.py).
It then writes a selection containing the source, effective filters, accepted
records, and annotation threshold.

Raw provider metadata and normalized enrichment remain nested under the source
record's `metadata`. The `SUMMARY` presentation row is not persisted.

### Materialization and annotation persistence

[`materialize_selection()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/download.py)
reads the fixed selection. For each accepted ID it:

```text
download_dataset(dataset_id)
  -> validate the signed-link response
  -> SMDataset.download_to_dir()
  -> validated local pair

get_annotations(dataset_id, options)
  -> results per molecular database and FDR
  -> optional matching ion images
  -> canonical molecular rows and spatial links

write_annotation_csv_pair()
  -> annotations.csv, pixel_intensities.csv beside the imzML pair

working-catalog upsert
  -> local source path and materialization status only;
     no annotation rows are written here
```

Download does not repeat discovery and cannot silently change the reviewed
cohort. It also does not import annotations into any SQLite catalog — that
happens only during
[composition](../../how-to/dataset-management/composing-a-cohort.md), through
`import_local_dataset()`. Annotation normalization and SQLite table
relationships are described in
[Annotation normalization](annotation-normalization.md) and
[SQLite catalogue](sqlite-catalog.md).

### Compatibility constraints

The deployed GraphQL schema must be treated as authoritative for the pinned
client version. In particular, `DatasetDiagnostic` in the used API exposes
`type` and `data` but not necessarily an `error` field. Queries request only
fields verified against the pinned/deployed interface.

METASPACE upstream `master` may contain fields not yet available in the
deployed endpoint. New fields require a real API probe and an offline
regression fixture before they are added to discovery.
