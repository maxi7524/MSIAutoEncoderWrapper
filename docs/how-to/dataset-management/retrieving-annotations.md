# Retrieve dataset annotations

Annotation retrieval materializes molecular results and, optionally, their
pixel links after a dataset selection has been discovered. Discovery statistics
are described in [Discover external datasets](discovering-datasets.md); the
provider calls and normalization stages are documented in
[METASPACE provider internals](../../library-internals/dataset-management/metaspace-provider.md).

## Choose the representation

METASPACE exposes dataset-level molecular annotations and first-isotope ion
images. The wrapper stores these as two related canonical representations:

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
annotations. The ambiguous legacy key `fdr` is rejected.

With `include_spatial=false`, retrieval stores molecular rows without fetching
ion images. This is faster, but the resulting catalogue cannot answer which
spectra are annotated. It must not be used to infer annotated or unannotated
pixel counts.

With `include_spatial=true`, ion images are transient inputs. They are not
saved under provider filenames or cached as individual image files. The
normalizer converts their non-zero positions into spectrum annotations stored
in `datasets/catalog.sqlite`. A repeated or restarted retrieval therefore asks
METASPACE for all qualifying images again.

## Understand discovery statistics versus persisted annotations

`include_molecule_stats` and `include_spatial_annotation_stats` are discovery
options. They enrich the in-memory result table but do not replace annotation
materialization:

- `molecule_count` counts distinct `(sumFormula, adduct)` identities at the
  selected FDR;
- `unique_molecule_count` is cohort-relative and can change when the selected
  datasets change;
- spatial discovery statistics union non-zero ion-image positions to report
  coverage, but do not persist all spectrum-to-molecule relations.

Retrieval reads the selected datasets again and writes normalized records to
the catalogue. This separation keeps a lightweight selection query distinct
from durable annotation data.

## Read normalized annotations

```python
from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader

reader = SQLiteAnnotationReader(
    "data/tutorial_workspace/datasets/catalog.sqlite",
    source="metaspace",
    dataset_id="dataset-id",
)

molecules = reader.get_annotations({"max_fdr": 0.1})
first_spectrum_molecules = reader.get_spectrum_annotations(0)
```

The reader is provider-independent because METASPACE fields are normalized
before insertion. Dataset identity remains a compound of `source` and
`dataset_id`; do not join records by dataset ID alone when a catalogue contains
multiple providers.

## Validate retrieval

After materialization, verify all three levels independently:

1. the selected dataset IDs exist in the catalogue;
2. dataset annotations are present at the expected maximum FDR;
3. when `include_spatial=true`, spectrum annotations exist and reference valid
   spectrum IDs for the materialized dataset.

An empty spatial result does not imply biological background. It can also mean
that no annotation passed the FDR threshold, an ion image was unavailable, or
spatial retrieval was disabled.

For the upstream operations used by the adapter, see the official METASPACE
[molecular annotation example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-dataset-metadata.html)
and [isotopic image example](https://metaspace2020.readthedocs.io/en/latest/content/examples/fetch-isotopic-images.html).
