# Retrieve dataset annotations

Annotation retrieval converts provider molecular results and optional ion
images into the canonical dataset and spectrum annotation schema.

## Purpose and available operations

### Dataset and spatial annotations

Dataset-level rows describe molecules. Spatial annotation adds spectrum IDs and
per-spectrum values derived from provider ion images. Merge selection requires
spatial links when annotated pixels determine inclusion.

### FDR consistency

The annotation FDR used during retrieval must match the discovery selection.
The canonical key is `annotation_fdr`; the ambiguous legacy key `fdr` is
rejected in retrieval configuration.

## Detailed instructions

### Configure retrieval

```json
{
  "annotation_fdr": 0.1,
  "include_spatial": true
}
```

`include_spatial=false` stores molecular rows without pixel links. Do not use
that result to infer annotated and unannotated spectra.

### Read normalized results

```python
from msi_autoencoder_wrapper.annotations import SQLiteAnnotationReader

reader = SQLiteAnnotationReader(
    "data/tutorial_workspace/datasets/catalog.sqlite",
    source="metaspace",
    dataset_id="dataset-id",
)
molecules = reader.get_annotations({"max_fdr": 0.1})
pixel_molecules = reader.get_spectrum_annotations(0)
```

The SQLite reader is provider-independent because provider output is normalized
before insertion.
