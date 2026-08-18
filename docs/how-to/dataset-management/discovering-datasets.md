# Discover external datasets

```{admonition} METASPACE API access
:class: warning

As of this writing, METASPACE does not allow this project to query datasets
through its public API. This guide documents the intended discovery workflow
for when API access is available.
```

Dataset discovery queries a registered provider, normalizes provider metadata,
and returns accepted and rejected records without downloading imzML or ibd
files. This guide documents the user-facing discovery interface. The internal
METASPACE request and normalization flow is described in
[METASPACE provider internals](../../library-internals/dataset-management/metaspace-provider.md).

## Create an explorer

`DatasetExplorer` stores the active filters, accepted records, manual
exclusions, and rejection diagnostics for one provider.

```python
from msi_dataset_manager.exploration import DatasetExplorer

explorer = DatasetExplorer(
    source="metaspace",
    cache_dir="assets/local/datasets/metaspace",
    refresh_cache=False,
)
```

`cache_dir` stores the METASPACE catalogue as `available-datasets.json`. The
catalogue supplies available-value tables and preselects dataset IDs for local
biological filters. `filter()` still contacts METASPACE for the selected IDs so
that annotation counts, current configuration, file sizes, and diagnostics are
not read from a stale catalogue.

Set `refresh_cache=True` to replace the catalogue before discovery. Leave it
disabled for repeated searches over the same catalogue.

## Inspect supported filters

```python
available = explorer.get_available_filters()
values = explorer.get_available_values("condition")
```

The filter schema identifies the expected type, default, exact argument passed
to the official client, and whether the wrapper applies the filter locally.
Unknown keys raise `ValueError` instead of being forwarded as arbitrary GraphQL
fields.

`get_available_values()` returns `value`, `label`, `count`, and `variants`.
The `variants` column lists raw METASPACE spellings combined into one declared
value. Canonical values use PascalCase. For example, `Mouse`, `Mus musculus`,
and `Mus musculus (mouse)` select the complete `Mouse` group. The mappings are
explicit; unlisted values are not merged by fuzzy or substring matching.

The filters `organism`, `organism_part`, `condition`, and
`growth_conditions` are applied locally because submitters enter these fields
as free text. A string selects one declared group. A list combines groups with
logical OR.

## Filter datasets

```python
filters = {
    "organism": "Mouse",
    "organism_part": "Kidney",
    "polarity": "Negative",
    "condition": "Wildtype",
    "mz_min": 200,
    "mz_max": 900,
    "annotation_fdr": 0.1,
    "min_annotation_count": 1,
    "include_molecule_stats": True,
    "exclude_dataset_ids": [],
}

results = explorer.filter(filters, summarise=True)
```

METASPACE-native filters such as `polarity`, `ionisation_source`, and
`analyzer_type` are passed to `SMInstance.datasets()`. Local free-text filters
first restrict the cached catalogue. The resulting IDs are then sent to
METASPACE together with the native filters. This avoids retrieving the full
provider catalogue for every search.

`exclude_dataset_ids` and the `mz_min`/`mz_max` coverage requirement are
applied together, immediately after metadata and m/z ranges are attached and
before annotation counts, molecular statistics, or spatial statistics are
requested. A dataset covers the requested range when its own `mz_min` is no
greater than the requested minimum and its `mz_max` is no less than the
requested maximum; datasets without both values, or with a narrower range, are
rejected with a diagnostic before any further API cost is spent on them.
`mz_min`/`mz_max` is a convenience pair for `mz_range: {"min": ..., "max": ...,
"mode": "covers"}`; supplying both forms together raises `ValueError`.

Discovery always requests current annotation counts and the following public
METASPACE metadata:

- annotation tolerance from `configJson.image_generation.ppm`;
- analysis version from `configJson.analysis_version`;
- acquired pixel count from `acquisitionGeometry.pixel_count`;
- imzML and ibd sizes from `sizeHash`;
- minimum and maximum observed m/z from the `IMZML_METADATA` diagnostic;
- instrument, analyzer, resolving power, ionization source, polarity, sample
  condition, and biological metadata.

The returned table contains separate `condition` and `diseases` columns.
`condition` is the submitter-provided experimental or biological condition and
may contain values such as `Wildtype`, `Control`, or `Frozen`. `diseases` is
reserved for explicit disease metadata and is not inferred from `condition`.

## Explore m/z coverage interactively

Two methods let a review session narrow a range before it is committed to
`filters`.

```python
coverage = explorer.count_mz_range_coverage(
    lower_bounds=[100, 200, 300, 400],
    upper_bounds=[700, 800, 900, 1000],
)
```

`count_mz_range_coverage()` returns a long table with `lower_bound`,
`upper_bound`, `range_width`, and `dataset_count` for every combination of the
supplied bounds (pairs where `lower_bound >= upper_bound` are skipped). It
counts datasets from the most recent `filter()`/`search()` result; pivot the
table with `pandas.DataFrame.pivot()` for a matrix view.

```python
matching = explorer.select_mz_range(min_mz=200, max_mz=900)
```

`select_mz_range()` previews datasets from the current result that cover
`[min_mz, max_mz]`; it does not modify the active filters or exclusions. To
have m/z coverage enforced during the provider query itself — before
annotation counts or molecular statistics are requested — pass
`mz_min`/`mz_max` directly in the `filters` mapping instead, as shown above.

## Calculate molecular statistics

Set `include_molecule_stats=True` to populate:

- `molecule_count`: distinct `(sumFormula, adduct)` identities in one dataset;
- `unique_molecule_count`: identities occurring only in that dataset within
  the current accepted cohort;
- `unique_molecules`: the corresponding formula-adduct labels.

The calculation retrieves annotation identities at `annotation_fdr`. It does
not download ion images. `min_molecule_count` and
`min_unique_molecule_count` enable the same calculation automatically before
applying their thresholds.

Set `include_spatial_annotation_stats=True` to download the first-isotope ion
images for annotations satisfying `annotation_fdr`. A pixel is counted once if
at least one selected image contains a finite, non-zero value at that position.
The result includes annotated and unannotated pixel counts, their fraction, the
number of contributing annotation images and databases, and a completion
status.

During discovery, one persistent progress bar reports all planned wrapper
operations. A transient second bar names the current operation. For spatial
retrieval, the persistent bar also reports the number of datasets and ion
images to retrieve. The official client's additional nested progress bar is
suppressed.

## Interpret the result table

Each ordinary row represents one accepted dataset. Relevant column groups are:

- identity: `dataset_id`, `name`, `project_url`, `submitter`, `group`,
  `projects`;
- biology: `organisms`, `organism_parts`, `condition`,
  `growth_conditions`, `diseases`;
- acquisition: `instruments`, `ionisation_source`, `analyzer_type`,
  `analyzer_resolving_power`, `polarity`, `image_size`, `pixel_count`;
- source size and range: `total_size_bytes`, `download_size_bytes`, `mz_min`,
  `mz_max`, `mz_tolerance_ppm`, `analysis_method`;
- annotations: `databases`, `annotation_count`, `molecule_count`,
  `unique_molecule_count`, `unique_molecules`;
- spatial annotations: `annotated_pixel_count`, `unannotated_pixel_count`,
  `annotated_pixel_fraction`, `spatial_annotation_count`,
  `spatial_annotation_database_count`, `spatial_stats_status`;
- review state: `excluded`.

Missing values mean that METASPACE did not expose the field or that the
corresponding optional calculation was not requested. A zero molecular count
is different from a missing molecular count.

With `summarise=True`, the final row has `dataset_id == "SUMMARY"`. It reports:

- the sum of known dataset sizes and whether size coverage is complete;
- the intersection of all dataset m/z ranges;
- the union of METASPACE analysis methods;
- the union of ionization sources and analyzer types.

The m/z intersection is defined only when every selected dataset provides
`mz_min` and `mz_max` and the ranges overlap.

## Inspect rejections and full metadata

```python
rejected = explorer.rejected()

dataset_id = results.loc[results["dataset_id"] != "SUMMARY", "dataset_id"].iloc[0]
record = explorer.source.get_dataset_metadata(dataset_id)
```

`rejected()` explains local free-text, m/z-coverage, and quantitative
rejections. Datasets removed directly by a native METASPACE filter are not
returned by the provider and therefore cannot have wrapper-generated rejection
reasons.

`get_dataset_metadata()` returns the normalized source record together with
the original METASPACE metadata and current GraphQL enrichment fields.

## Query through the dataset CLI

```bash
msi-datasets query \
  --source metaspace \
  --workspace-path workspace \
  --filters workspace/configs/datasets/kidney/filter.json \
  --selection workspace/configs/datasets/kidney/selection.json
```

The query stores accepted discovery records in the catalogue and writes a
selection snapshot. Download and annotation materialization are separate
operations described in [Download datasets](downloading-datasets.md) and
[Retrieve dataset annotations](retrieving-annotations.md). See
[Use the msi-datasets CLI](command-line-workflow.md) for every argument.

## Diagnose common failures

- `Unknown METASPACE filters` indicates a misspelled or unsupported public
  filter key.
- `Use either mz_range or mz_min/mz_max, not both` and `mz_min and mz_max must
  be provided together` indicate an invalid m/z filter combination.
- A GraphQL error occurs before local filtering and means the installed client
  and deployed schema disagree or access to a requested field was denied.
- Empty `mz_min` and `mz_max` values mean the dataset has no accessible
  `IMZML_METADATA` diagnostic; they must not be inferred from annotation m/z.
- Spatial statistics can fail when ion-image dimensions disagree, an
  annotation has no matching ion image, or annotated pixels exceed acquisition
  geometry.
- Private datasets require credentials configured for the official METASPACE
  client.

The official interfaces used by the adapter are documented in the
[METASPACE Python client API](https://metaspace2020.readthedocs.io/en/latest/content/apireference/sm_annotation_utils.html)
and the [dataset metadata example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-dataset-metadata.html).
