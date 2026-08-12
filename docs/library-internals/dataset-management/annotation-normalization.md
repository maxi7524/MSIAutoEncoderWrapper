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

[`normalize_spectrum_annotations()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/normalization/spatial_annotations.py)
maps one-based imzML coordinates to zero-based provider image indices. Finite
positive pixels create spectrum links. It has no current caller in this
package: `download`'s CSV writer performs the equivalent coordinate mapping
inline (see
[`write_annotation_csv_pair()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/operations/annotation_csv.py))
instead of calling it. Provider ion images are retrieved as described in
[METASPACE provider](metaspace-provider.md#ion-images).

### Normalize paired CSV

[`read_canonical_csv_annotations()`](../../../packages/msi_dataset_manager/src/msi_dataset_manager/annotations/csv.py)
matches molecular and intensity rows by decimal m/z, formula, and adduct, and
maps `x*_y*` columns to reader spectrum positions. `import_local_dataset()`
uses this function.

`msi_autoencoder_wrapper` reads the same paired CSV layout directly — without
importing it into a catalog — through
[`read_metaspace_csv_annotations()`](../../../src/msi_autoencoder_wrapper/annotations/strategies/metaspace_csv_annotation_reader.py).
The two functions apply the same matching algorithm but are separate
implementations in separate, mutually independent distributions; they are not
literally shared code, and a format change must be applied to both.
