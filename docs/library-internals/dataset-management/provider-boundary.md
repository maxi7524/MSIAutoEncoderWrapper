# Dataset provider boundary

Provider adapters translate external APIs into one discovery, metadata,
annotation, and download contract.

## General abstraction

### Shared contract

`DatasetSource` defines filter discovery, search, diagnostics, metadata,
annotations, and file download. Operations depend on this contract and do not
parse provider payloads.

### Secret boundary

Authentication values remain environment or runtime options and are excluded
from exported source configuration.

## Detailed implementation

### Registration and resolution

[`DatasetSourceManager`](../../../src/msi_autoencoder_wrapper/dataset_management/sources/source_manager.py)
discovers source strategies and resolves keys, compatible classes, or instances.

### Built-in adapters

[`metaspace.py`](../../../src/msi_autoencoder_wrapper/dataset_management/sources/strategies/metaspace.py)
owns METASPACE catalog, result, ion-image, and download interpretation.
[`pride.py`](../../../src/msi_autoencoder_wrapper/dataset_management/sources/strategies/pride.py)
owns PRIDE project and file APIs. Unsupported provider capabilities return
explicit diagnostics or errors rather than fabricated common fields.
