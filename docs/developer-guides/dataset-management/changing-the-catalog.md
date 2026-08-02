# Change the SQLite catalog

Catalog changes affect provider operations, annotation readers, model targets,
merged provenance, and persisted workspaces.

## Catalog scope and invariants

### Identity invariants

Source identity is `(source, dataset_id)`. Merged identity is
`merged_dataset_id`. Every merged index maps to at most one source spectrum.

### Atomic replacement

Annotation and spectrum-mapping replacement must remain transactional.

## Implementation instructions

### Design a schema migration

Increment and validate an explicit schema version before incompatible table
changes. Define migration or intentionally reject older catalogs; do not infer
columns opportunistically at read time.

### Update all consumers

Change catalog methods, SQLite annotation reader, import/download operations,
merge, configuration tests, and documentation together. Preserve indexed lookup
for dataset and spectrum annotations.

### Test persistence

Create, close, reopen, query, replace, merge, remove source files, and verify
merged annotation access. Include transaction rollback on invalid input.
