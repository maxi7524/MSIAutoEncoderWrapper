# Targets and annotations

Annotations become model targets only through explicit dataset target
specifications and availability masks.

## General abstraction

### Annotation and target distinction

Annotation readers expose metadata and molecules. Datasets interpret selected
fields as single-label or multi-label targets. Models consume target tensors,
not annotation reader objects.

### Mask semantics

Every target has an availability mask. Missing metadata does not become an
arbitrary class and unavailable labels do not contribute to masked losses.

## Detailed implementation

### Build mappings

[`class_assignment.py`](../../../src/msi_autoencoder_wrapper/models/datasets/class_assignment.py)
creates deterministic semantic-value to class-index mappings unless an explicit
mapping is provided. Molecules use formula/adduct keys.

### Build samples and batches

`PixelDataset` reads dataset or spectrum metadata, creates target tensors and
masks, and caches class mappings. `RawSpectrumCollator` stacks values and masks
into `TargetBatch`; schemas remain shared metadata.

### Preserve cohort consistency

`CohortDataset` compares target schemas across members. Inconsistent mappings
fail before training rather than silently changing class meaning by image.
