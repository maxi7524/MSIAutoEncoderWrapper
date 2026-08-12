# Compose a cohort dataset

Composition merges every canonical local dataset in a cohort into one imzML
image and, in the same step, imports their annotation CSVs into a
self-contained SQLite catalog and builds a cohort-wide molecule-occurrence
index. It operates entirely on canonical local datasets — already downloaded
(see [Download selected datasets](downloading-datasets.md)), or placed
manually in the same layout — and does not contact a provider. It is now the
only supported path from local imzML/ibd pairs and their annotation CSVs to a
merged, catalog-backed cohort dataset: standalone `merge` and `import-local`
CLI commands do not exist.

## Purpose and available operations

### Input datasets

Each `dataset_id` must have a complete local imzML/ibd pair under
`workspace/datasets/<dataset_id>/`. A dataset missing that pair is skipped —
not an error — and recorded in the output's `missing_dataset_ids`; composition
proceeds with the remaining datasets. A dataset with a complete paired
annotation CSV — canonical `annotations.csv` + `pixel_intensities.csv`, the
files [Download selected datasets](downloading-datasets.md) writes, with a
legacy `metaspace_annotations.csv` + `*_pixel_intensities.csv` pair also
accepted — is imported into the cohort catalog first; one without annotations
is still merged, recorded with status `materialized_without_annotations`.

### What composition produces

One call writes three artifacts, all under `workspace/datasets/<cohort_id>/`:

- `<cohort_id>.imzML` (+ `.ibd`) — the merged image;
- `<cohort_id>.sqlite` — the **composed catalog**: a self-contained SQLite
  catalog holding this cohort's imported annotations and merged spectrum
  mappings, colocated beside the merged image. Because it sits next to the
  image under the same stem, `msi_autoencoder_wrapper`'s automatic annotation
  detection finds it without any explicit configuration — see
  [Attach molecular annotations](../data-input-and-preprocessing/annotations.md#supported-sources-and-selection-priority).
  This is a different file from the **working catalog** used by
  [`query`/`download`](downloading-datasets.md#choose-the-cohort-catalog)
  under `workspace/configs/datasets/<cohort_id>/<cohort_id>.sqlite`, which only
  tracks file materialization state and holds no imported annotations;
- `composition.json` — the normalized composition configuration actually used
  (including which requested dataset IDs were available versus missing), for
  provenance and reruns;
- `annotation_index.json` — the cohort-wide molecule-occurrence index and
  masks, described below.

### Molecule occurrence masks

After merging, the operation reads every input dataset's stored annotations
(at or below `max_fdr`, if given) from the composed catalog and groups them by
`(formula, adduct, database_name, database_version)`. For each distinct
molecule it records every dataset that observed it and that observation's FDR,
then derives:

- `dataset_occurrence_count` and `occurrence_mask` — how many, and which, of
  the cohort's datasets observed the molecule (mask entries follow the order
  of the resolved `dataset_ids`);
- `single_dataset_only` — `true` when the molecule was observed in exactly one
  dataset;
- `selected` — `true` when `dataset_occurrence_count >= minimum_dataset_occurrence`.

No molecule is removed from the index for occurring in only one dataset; a
single-dataset molecule is less safe to extrapolate onto other images in the
cohort but is still reported, with `single_dataset_only=true`, so downstream
analysis or training can choose to keep or drop it. `masks` in the output
collects three reusable boolean arrays over the full molecule list —
`all`, `single_dataset`, and `minimum_dataset_occurrence` — aligned with
`molecules`, so a consumer can select a mask without re-deriving it from
`molecules` directly.

`max_fdr` and `minimum_dataset_occurrence` apply identically to spectrum
selection during merge (which spectra count as "annotated" for
`unannotated_ratio`/`unannotated_amount` sampling, below) and to this molecule
index.

## Detailed instructions

### Compose through Python

```python
from msi_dataset_manager.operations import compose_cohort

output = compose_cohort(
    workspace_path="workspace",
    cohort_id="kidney",
    source="metaspace",
    dataset_ids=["dataset-a", "dataset-b", "dataset-c"],
    row_width=128,
    max_fdr=0.1,
    minimum_dataset_occurrence=2,
    unannotated_ratio=1.0,
    random_seed=0,
)
```

### Compose through the CLI

```bash
msi-datasets compose \
  --workspace-path workspace \
  --cohort-id kidney \
  --source metaspace \
  --selection workspace/configs/datasets/kidney/selection.json \
  --row-width 128 \
  --max-fdr 0.1 \
  --minimum-dataset-occurrence 2 \
  --unannotated-ratio 1.0 \
  --random-seed 0
```

`--selection` supplies the dataset IDs to compose (every `dataset_id` in its
`datasets` array) without repeating them on the command line. Repeated
`--dataset-id` overrides both `--selection` and any `dataset_ids` inside
`--config`.

`--config CONFIG.json` accepts the same fields as the Python call
(`source`, `dataset_ids`, `row_width`, `max_fdr`, `minimum_dataset_occurrence`,
`unannotated_ratio`, `unannotated_amount`, `random_seed`), plus arbitrary extra
keys that are carried into `composition.json` as provenance. Where both are
given, a value inside `--config` takes precedence over the matching CLI flag
for every field except `--dataset-id`, which always wins.

### Read the cohort molecule index

```python
import json

annotation_index = json.loads(
    (Path("workspace/datasets/kidney") / "annotation_index.json").read_text()
)
selected_molecules = [
    molecule
    for molecule, selected in zip(
        annotation_index["molecules"], annotation_index["masks"]["minimum_dataset_occurrence"]
    )
    if selected
]
```

For spectrum-level annotation lookups on the composed image itself, use
[`SQLiteAnnotationReader`](retrieving-annotations.md#read-normalized-annotations)
against `workspace/datasets/kidney/kidney.sqlite` (the composed catalog) with
`merged_dataset_id="kidney"`.

## Building blocks composition uses internally

These are Python-only implementation details, not independently CLI-exposed;
most compositions never need to call them directly.

### Local CSV import

`compose_cohort()` imports each input's annotation CSV pair with
[`import_local_dataset()`](../../library-internals/dataset-management/annotation-normalization.md):

```python
from msi_dataset_manager.catalog import DatasetCatalog
from msi_dataset_manager.operations.import_local import import_local_dataset

catalog = DatasetCatalog("workspace/datasets/kidney/kidney.sqlite")
result = import_local_dataset(
    catalog=catalog,
    source="metaspace",
    dataset_id="example-1",
    name="Example 1",
    imzml_path="workspace/datasets/example-1/example-1.imzML",
    annotations_path="workspace/datasets/example-1/annotations.csv",
    pixel_intensities_path="workspace/datasets/example-1/pixel_intensities.csv",
    metadata={},
)
```

The result contains `spectra`, `annotations`, and `spatial_links` counts.
Mass-to-charge values must match uniquely between the two CSV files; negative
or non-finite intensities are rejected; zero and empty values create no
spatial link.

### Custom spectrum selection during merge

`compose_cohort()` always applies annotation-aware automatic spectrum
selection (all annotated spectra, plus optional sampled unannotated ones) to
every input. For a merge with an explicit, per-dataset spectrum-index list —
not exposed by `compose_cohort()` — call
[`ImzMLMerger`](../../library-internals/dataset-management/merge-and-provenance.md)
directly:

```python
from msi_dataset_manager.catalog import DatasetCatalog
from msi_dataset_manager.operations.merge import ImzMLMergeInput, ImzMLMerger

catalog = DatasetCatalog("workspace/datasets/kidney/kidney-pilot.sqlite")
ImzMLMerger(catalog).merge(
    inputs=[
        ImzMLMergeInput(
            source="metaspace",
            dataset_id="dataset-a",
            imzml_path="workspace/datasets/dataset-a/dataset-a.imzML",
            spectrum_ids=[0, 1, 2, 5],
        ),
    ],
    output_path="workspace/datasets/kidney-pilot/kidney-pilot.imzML",
    merged_dataset_id="kidney-pilot",
    row_width=128,
)
```

An input's `spectrum_ids`, when given, overrides automatic annotation-aware
selection entirely for that dataset with an explicit list of source spectrum
indices.
