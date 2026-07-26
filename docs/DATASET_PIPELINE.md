# External dataset pipeline

The external-data pipeline separates metadata discovery from file downloads.
METASPACE is the first adapter, but catalog, annotation-reader, and merge APIs do
not depend on its client types.

## Workspace layout

New workspaces use the following layout:

```text
workspace/
  datasets/
    catalog.sqlite
    manifests/
    sources/
      metaspace/
        <dataset_id>/
          <dataset_id>.imzML
          <dataset_id>.ibd
          metadata.json
          annotations.json
    merged/
      <merged_dataset_id>/
        dataset.imzML
        dataset.ibd
```

`workspace.get_imgs_dir()` remains as a compatibility alias for
`workspace.get_datasets_dir()`. A dataset stored in a new workspace is resolved
from `datasets/<dataset_name>/<dataset_name>.imzML`; existing flat files in the
configured legacy directory remain readable.

## Catalog stage

The catalog stage queries metadata and writes a reproducible JSON manifest. It
does not download imzML/ibd files.

```bash
python assets/scripts/datasets/manage_datasets.py catalog \
  --workspace /path/to/workspace \
  --source metaspace \
  --filters assets/configs/datasets/metaspace_filters.json \
  --manifest /path/to/workspace/datasets/manifests/candidates.json
```

The filter object is passed to the official METASPACE client's `datasets(...)`
method. Unsupported provider filters therefore fail visibly instead of being
silently ignored.

## Download stage

The download stage accepts a previously written manifest. It downloads the
imzML/ibd pair, provider metadata, and all annotations exposed by the requested
retrieval options.

```bash
python assets/scripts/datasets/manage_datasets.py download \
  --workspace /path/to/workspace \
  --source metaspace \
  --manifest /path/to/workspace/datasets/manifests/candidates.json \
  --annotation-options assets/configs/datasets/metaspace_annotations.json \
  --dataset-id <selected_dataset_id>
```

Retrieval options are not training filters. The broad METASPACE FDR level is
used to import available records. Experimental restrictions are supplied later
to an annotation reader, for example:

```python
wrapper.context_manager.set_annotation_reader(
    "CatalogAnnotationReader",
    catalog_path=wrapper.workspace.get_dataset_catalog_path(),
    source="metaspace",
    dataset_id="<dataset_id>",
    default_filters={"max_fdr": 0.1, "database_name": "HMDB"},
)
```

With no `default_filters`, the reader returns every imported annotation.

## Merge stage

Merge configuration explicitly selects source spectrum/spatial indices. The
result is a processed imzML/ibd pair with consecutive row-major coordinates.

```bash
python assets/scripts/datasets/manage_datasets.py merge \
  --workspace /path/to/workspace \
  --config assets/configs/datasets/merge.example.json
```

The SQLite `spectrum_mappings` table stores only the stable provenance required
by the current design:

```text
(source, source_dataset_id, source_spatial_id) -> merged_spectrum_index
```

Coordinates are owned by the source and merged imzML readers. Coordinate
transforms are deliberately not duplicated in the catalog.

For collections that do not fit twice on local storage, `download-merge` keeps
only one source dataset in `datasets/.staging` at a time:

```bash
python assets/scripts/datasets/manage_datasets.py download-merge \
  --workspace /path/to/workspace \
  --source metaspace \
  --manifest /path/to/workspace/datasets/manifests/candidates.json \
  --output /path/to/workspace/datasets/merged/pilot/dataset.imzML \
  --merged-dataset-id pilot \
  --row-width 128
```

Metadata, annotations, and index provenance remain in `catalog.sqlite`. Add
`--keep-downloads` only when the original source pairs should also remain in the
workspace.

## Annotation responsibility

Imported records are immutable source data. `CatalogAnnotationReader` applies
optional database, database-version, formula, adduct, and maximum-FDR filters at
read time. `MergedCatalogAnnotationReader` first maps the merged spectrum index
back to its source spatial ID and then exposes the corresponding source records.

The following choices intentionally remain outside the importer:

- molecule-head target semantics;
- conversion of ion-image intensity into a pixel target;
- condition vocabulary and multi-class versus multi-label encoding;
- production METASPACE dataset selection;
- train/validation/test grouping rules beyond the requirement to split by
  independent source dataset, patient, or experiment rather than pixels.

These decisions belong to the dataset/training configuration and can be added
without downloading the source data again.

`grouped_dataset_split(...)` already provides deterministic leakage-safe splits
from catalog records. The configured `group_fields` can point to patient or
experiment metadata. Records with missing grouping metadata are kept as separate
dataset groups instead of being incorrectly treated as one shared patient.

## Initial experiment matrix

The first controlled comparison contains four runs of the same autoencoder:

| Reconstruction criterion | Auxiliary heads |
|---|---|
| MSE | none |
| Masserstein | none |
| MSE | condition and molecule |
| Masserstein | condition and molecule |

Encoder, decoder, latent size, binner, normalization, split, seed, optimizer,
training budget, and existing loss weights must remain fixed. Head-enabled
configs should only be added after their target schema is finalized.
