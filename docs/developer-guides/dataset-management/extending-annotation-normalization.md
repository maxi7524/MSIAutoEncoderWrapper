# Extend annotation normalization

New provider annotation formats must map into canonical molecular rows and
optional spectrum links before SQLite readers consume them.

## Normalization scope and invariants

### Molecular identity

Preserve formula, adduct, m/z, FDR, database provenance, and provider raw data.
Use a stable annotation ID within the source dataset.

### Spatial identity

Spectrum links use source spectrum IDs and finite non-negative values. Coordinate
conventions must be translated explicitly.

## Implementation instructions

### Implement provider mapping

Keep provider parsing in the provider or a provider-specific normalization
module. Return the mapping accepted by `DatasetCatalog.replace_annotations()`.

### Handle incomplete spatial data

Decide whether the provider guarantees a complete molecular-to-image match. If
spatial retrieval was requested and a molecular result cannot be mapped, fail
rather than silently label a partial dataset as complete.

### Test direct and stored reads

Insert normalized records into a temporary catalog and compare dataset and
spectrum reader output. Test zero, negative, non-finite, duplicate, and missing
coordinate/molecule cases.
