# Annotation normalization

Annotation normalization converts provider or paired CSV structures into one
dataset-level molecule schema and many-to-many spectrum links.

## General abstraction

### Canonical records

Molecular records retain formula, adduct, m/z, FDR, database provenance, and raw
provider values. Spatial links attach annotation IDs to source spectrum IDs with
optional positive intensity.

### Reader independence

SQLite annotation readers query normalized records and therefore do not contain
METASPACE or PRIDE API logic.

## Detailed implementation

### Normalize provider ion images

[`normalize_spectrum_annotations()`](../../../src/msi_autoencoder_wrapper/dataset_management/normalization/spatial_annotations.py)
maps one-based imzML coordinates to zero-based provider image indices. Finite
positive pixels create spectrum links.

### Normalize paired CSV

[`read_metaspace_csv_annotations()`](../../../src/msi_autoencoder_wrapper/annotations/strategies/metaspace_csv_annotation_reader.py)
matches molecular and intensity rows by decimal m/z and maps `x*_y*` columns to
reader spectrum positions. Local import and direct CSV reading share this path.
