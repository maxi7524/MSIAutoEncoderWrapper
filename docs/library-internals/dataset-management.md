# Dataset management internals

This document describes how `msi_autoencoder_wrapper.dataset_management` discovers, materializes, normalizes, and merges external MSI datasets.

## Scope

This document covers internal responsibilities and contracts. User commands and configuration fields are described in [Dataset management protocol](../how-to/dataset-management.md). Instructions for implementing another provider are in [Adding a dataset source](../developer-guides/adding-a-dataset-source.md).

## Component boundaries

```text
DatasetExplorer
    -> DatasetSource.filter()
    -> reviewed filter JSON
    -> query_to_selection()
    -> selection JSON + DatasetCatalog metadata
    -> materialize_selection() or materialize_and_merge_selection()
       -> DatasetSource.download_dataset()
       -> DatasetSource.get_annotations()
       -> normalize_spectrum_annotations()
       -> DatasetCatalog annotations and spectrum links
       -> ImzMLMerger or streaming writer
       -> merged imzML/ibd + reversible spectrum mappings
```

`DatasetSource` owns provider communication and provider-record interpretation. `DatasetExplorer` presents the common discovery interface. Operations coordinate lifecycle stages without parsing provider payloads. `DatasetCatalog` stores the canonical local representation. Annotation readers consume that representation without contacting providers.

## Discovery artifacts

`query_to_selection()` removes `exclude_dataset_ids` before calling the provider, validates returned records, upserts metadata into SQLite, and writes a schema-versioned selection JSON. The selection is an immutable hand-off between review and materialization. It stores the effective filters so `annotation_fdr` can be reused during annotation retrieval.

METASPACE constructs an in-memory catalogue during source initialization. With `cache_dir=None`, the catalogue is not written to disk. With `cache_dir` set, `available-datasets.json` is loaded when valid or fetched and stored when absent. `refresh_cache=True` replaces the file from the service.

## METASPACE provider boundary

The adapter uses the pinned `metaspace2020` client and its [documented dataset interface](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html) to:

- call `SMInstance.datasets()` for native dataset filters;
- call GraphQL annotation aggregates for review statistics;
- call `SMDataset.results()` for molecular result rows;
- call `SMDataset.all_annotation_images()` for first-isotope spatial images;
- call `SMDataset.download_links()` or `download_to_dir()` for source files.

Local aggregate filters are applied after provider discovery. They include annotation counts, molecule counts, unique molecules within the current query result, spatial pixel statistics, optical-image presence, and explicit exclusions.

The source accepts `METASPACE_API_KEY` through `metaspace_client_options()`. `metaspace_session.sh` validates the environment value with `SMInstance.logged_in()` before download commands use it. The secret is not included in serialized source configuration.

## Annotation FDR contract

`annotation_fdr` is the external-result threshold used by:

- discovery annotation counts;
- optional molecule statistics;
- optional spatial annotation statistics;
- molecular result retrieval;
- ion-image retrieval.

The query selection stores the threshold. Download resolves annotation options against it and rejects a different threshold. Individual provider rows may retain their source `fdr` value in `raw_json` and the canonical `annotations.fdr` column; those values describe individual results, while `annotation_fdr` describes the retrieval boundary.

## Spatial normalization

For each database, METASPACE results and ion-image groups are keyed by `(sumFormula, adduct)`. When spatial retrieval is enabled, every result must have a matching ion image. A missing match stops materialization because the molecular result cannot be assigned completely to pixels.

`normalize_spectrum_annotations()` iterates the source imzML spectra and converts one-based `(x, y)` coordinates to zero-based ion-image indices. A finite positive image value creates a link containing `annotation_id`, `spectrum_id`, and the preserved intensity. Intensity magnitude does not change whether the link exists.

## Canonical SQLite state

`catalog.sqlite` contains:

- `datasets`: source identity, metadata, local path, and lifecycle status;
- `annotations`: molecule identity, database provenance, result FDR, and normalized raw data;
- `spectrum_annotations`: many-to-many molecule-to-source-spectrum links;
- `merged_datasets`: merged imzML artifacts;
- `spectrum_mappings`: merged index to source dataset and source spectrum ID.

Replacing a source dataset's annotations is atomic. The merged reader resolves `merged_spectrum_index` through `spectrum_mappings` and then queries the original `spectrum_annotations` rows.

## Download lifecycle

Materialization writes sources under `datasets/sources/<source>/<dataset_id>`. METASPACE first reuses a complete non-empty canonical pair. Signed-link downloads write each file to a `.part` path and replace the final path after a non-empty response completes. Unsupported files, quota sentinel files, and incomplete pair responses are rejected before transfer.

`materialize_and_merge_selection()` uses `datasets/.staging/<source>/<dataset_id>`. It downloads and normalizes one source, appends selected spectra, and removes staging unless `keep_downloads` is enabled. A failed run removes only partial merged output files.

## Merge selection and provenance

Without explicit `spectrum_ids`, all source spectra present in `spectrum_annotations` are selected. Optional unannotated sampling is deterministic per source dataset because the seed namespace contains the source and dataset ID. Sampling is therefore per dataset, not global across the merged result.

The output writer assigns consecutive rectangular coordinates using `row_width`. Original spatial geometry is not retained as output geometry. Source identity and source spectrum ID remain in imzML user parameters and `spectrum_mappings`.
