# Adding a dataset source

This guide describes how to integrate another external MSI database with `dataset_management`.

## Scope

The guide covers the source adapter contract, canonical artifacts, registration, error handling, and tests. Existing component relationships are described in [Dataset management internals](../library-internals/dataset-management.md).

## Implement the source contract

Create a strategy in `src/msi_autoencoder_wrapper/dataset_management/sources/strategies`. Subclass `DatasetSource` and register the class:

```python
from ..base import DatasetSource
from ..source_manager import DatasetSourceManager


@DatasetSourceManager.register_source("provider-key")
class ProviderDatasetSource(DatasetSource):
    source_name = "provider-key"
```

Implement:

- `get_available_filters()` for notebook-readable filter definitions;
- `get_available_values(filter_key)` for enumerable provider metadata;
- `filter(filters)` for canonical candidate records;
- `get_accepted_records()` and `get_rejected_records()` for review state;
- `get_dataset_metadata(dataset_id)` for the complete provider metadata;
- `download_dataset(dataset_id, destination)` for one canonical imzML/ibd pair;
- `get_annotations(dataset_id, options)` for provider molecular annotations.

Public methods must use Sphinx-compatible reStructuredText docstrings. Provider I/O and transformations must log major stages through `get_custom_logger(__name__)` without logging credentials or full datasets.

## Produce canonical discovery records

Every accepted record must contain:

```python
{
    "dataset_id": "stable-provider-id",
    "name": "display name",
    "source": "provider-key",
    "metadata": {},
}
```

Preserve provider metadata inside `metadata`. Add stable links used during manual review. Reject incomplete pairs, ambiguous pair assignments, unsupported annotation formats, and records that cannot satisfy the provider's documented metadata contract. Rejected records must include a reason and a provider page when available.

Separate native filters from local aggregate filters. Do not send library-only keys such as `exclude_dataset_ids` to the provider.

## Produce one MSI image per dataset record

`download_dataset()` must create:

```text
<destination>/<dataset_id>.imzML
<destination>/<dataset_id>.ibd
```

Treat each complete pair as one image. Do not combine several provider acquisitions inside the source adapter. Reuse an existing complete non-empty pair. For network downloads, use temporary files and publish the final path only after transfer validation.

Translate provider quota or access failures into project exception types. Do not retry automatically unless a retry policy is added to the public configuration and documented.

## Normalize molecular annotations

Return one mapping per provider molecular result. Preserve at least:

- a stable annotation identifier;
- formula or other molecular identity supported by the provider;
- adduct when available;
- source database and version;
- result FDR when available;
- provider fields needed for later inspection.

Only create `spectrum_ids` from a provider-supplied, curated spatial annotation. Do not infer molecule locations from local spectral peaks. A source may either return explicit `spectrum_ids` or an `ion_image` aligned with the imzML coordinates. If a requested spatial result cannot be matched, report or reject it according to the source contract instead of silently assigning a location.

## Register the strategy

Import the strategy in `sources/strategies/__init__.py` if discovery requires an explicit module import. Confirm that:

```python
DatasetSourceManager.discover_strategies()
source = DatasetSourceManager.get_source("provider-key")
```

constructs the adapter with documented keyword options.

Do not add provider parsing to `SQLiteAnnotationReader`, merge operations, or model datasets. Those components consume the canonical source contract.

## Add tests

Tests must not depend on downloading a live external dataset. Provide an injected fake client and copy the repository's compact imzML/ibd fixture into a temporary destination.

Cover:

1. filter mapping and accepted/rejected records;
2. available filter values;
3. metadata preservation;
4. complete-pair detection and reuse;
5. unsupported or incomplete provider responses;
6. annotation identity and spatial links;
7. selection serialization;
8. mocked materialization into the workspace;
9. merge provenance from a merged spectrum back to its source annotations;
10. provider authentication and quota errors when applicable.

Run focused source, pipeline, merger, catalog, and annotation-reader tests before the full suite.
