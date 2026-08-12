# Retrieve dataset annotations

```{admonition} METASPACE API access
:class: warning

As of this writing, METASPACE does not allow this project to retrieve
annotations through its public API. This guide documents the intended
retrieval workflow for when API access is available. Annotations already
imported into a composed catalog remain fully queryable; see
[Read normalized annotations](#read-normalized-annotations) below and
[Inspect the catalog](inspecting-the-catalog.md).
```

Annotation retrieval is part of [`download`](downloading-datasets.md): after a
dataset selection has been discovered, it materializes molecular results and,
optionally, their pixel links as CSV files beside the source image. Those CSV
files are only imported into a queryable SQLite catalog later, during
[`compose`](composing-a-cohort.md). Discovery statistics are described in
[Discover external datasets](discovering-datasets.md); the provider calls and
normalization stages are documented in
[METASPACE provider internals](../../library-internals/dataset-management/metaspace-provider.md).

## Choose the representation

METASPACE exposes dataset-level molecular annotations and first-isotope ion
images. The wrapper stores these as two related canonical representations,
once imported:

- dataset annotations identify a molecule, adduct, database, and FDR for one
  dataset;
- spectrum annotations link the normalized molecule to acquired spectrum IDs
  derived from its ion image.

Dataset annotations are sufficient for formula/adduct analyses. Spectrum
annotations are required when downstream selection or analysis depends on
which pixels are annotated.

## Configure retrieval

```json
{
  "annotation_fdr": 0.1,
  "include_spatial": true
}
```

`annotation_fdr` is the maximum FDR accepted from METASPACE. Use the same value
as during discovery when discovery counts must agree with the materialized
annotations; [Download datasets](downloading-datasets.md) rejects a mismatch
between a selection's stored `annotation_fdr` and an explicitly supplied one.
The ambiguous legacy key `fdr` is rejected.

With `include_spatial=false`, retrieval writes molecular rows without fetching
ion images. This is faster, but the resulting CSV pair cannot answer which
spectra are annotated. It must not be used to infer annotated or unannotated
pixel counts.

With `include_spatial=true`, ion images are transient inputs. They are
converted directly into `pixel_intensities.csv` columns and are not saved
under provider filenames or cached as individual image files. A repeated or
restarted retrieval — when the CSV pair is not already complete — therefore
asks METASPACE for all qualifying images again.

## Understand discovery statistics versus retrieved CSVs

`include_molecule_stats` and `include_spatial_annotation_stats` are discovery
options. They enrich the in-memory result table but do not replace annotation
retrieval:

- `molecule_count` counts distinct `(sumFormula, adduct)` identities at the
  selected FDR;
- `unique_molecule_count` is cohort-relative and can change when the selected
  datasets change;
- spatial discovery statistics union non-zero ion-image positions to report
  coverage, but do not write any file.

Retrieval reads the selected datasets again and writes the CSV pair. This
separation keeps a lightweight selection query distinct from durable
annotation data.

## Read normalized annotations

Reading requires a **composed catalog** — the one [`compose`](composing-a-cohort.md)
writes at `workspace/datasets/<cohort_id>/<cohort_id>.sqlite` after importing
the CSV pairs. A dataset that has only been downloaded, and not yet composed,
has no annotations queryable this way; read its `annotations.csv`/
`pixel_intensities.csv` directly, or run `compose` first.

```python
from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader

reader = SQLiteAnnotationReader(
    "workspace/datasets/kidney/kidney.sqlite",
    source="metaspace",
    dataset_id="dataset-id",
)

molecules = reader.get_annotations({"max_fdr": 0.1})
first_spectrum_molecules = reader.get_spectrum_annotations(0)
```

`SQLiteAnnotationReader` lives in `msi_autoencoder_wrapper` and reads the
catalog `msi_dataset_manager` writes; it is provider-independent because
METASPACE fields are normalized before insertion. Dataset identity remains a
compound of `source` and `dataset_id`; do not join records by dataset ID alone
when a catalogue contains multiple providers.

## Validate retrieval

After `download`, verify:

1. `annotations.csv` and `pixel_intensities.csv` exist and are non-empty for
   every dataset that should have annotations;
2. the materialization report (`materialization.json`) shows
   `annotation_statuses[dataset_id]` as `"downloaded"` or `"reused"`, not a
   `"failed: ..."` entry.

After `compose`, verify the composed catalog independently: the selected
dataset IDs exist in it, dataset annotations are present at the expected
maximum FDR, and — when the source CSVs included spatial links — spectrum
annotations exist and reference valid spectrum IDs.

An empty spatial result does not imply biological background. It can also mean
that no annotation passed the FDR threshold, an ion image was unavailable, or
spatial retrieval was disabled.

For the upstream operations used by the adapter, see the official METASPACE
[molecular annotation example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-dataset-metadata.html)
and [isotopic image example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-isotopic-images.html).
