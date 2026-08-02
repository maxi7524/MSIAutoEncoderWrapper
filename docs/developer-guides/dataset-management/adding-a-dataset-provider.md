# Add a dataset provider

A provider adapter owns external API semantics and implements the shared local
dataset-source contract.

## Provider scope and required operations

### Required methods

Implement available filters/values, search, accepted/rejected diagnostics,
metadata retrieval, annotation retrieval, and dataset download. Unsupported
capabilities must be explicit.

### External boundary

Provider payload parsing, pagination, authentication, quota handling, and file
selection remain inside the adapter. Operations consume normalized records.

## Implementation instructions

### Implement and register

Create a strategy under `dataset_management/sources/strategies`, inherit
`DatasetSource`, and decorate it with `DatasetSourceManager.register_source()`.

### Normalize records

Return stable source/dataset identity, name, metadata, and rejection reasons.
Never serialize secrets. Validate downloaded imzML/ibd pairs before reporting
materialization.

### Add tests

Mock the provider boundary. Test discovery, pagination, filtering, diagnostics,
metadata, annotation options, download reuse/failure, authentication, quota,
registration, and end-to-end materialization into a temporary catalog.
