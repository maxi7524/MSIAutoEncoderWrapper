# Workspace internals

Workspace internals translate logical image/model identities into filesystem
paths and atomic artifact operations.

## General abstraction

### Layout ownership

The workspace owns directory names and active filesystem identities. Components
store portable paths and configurations but do not construct workspace layouts.

### Model-store ownership

`ModelStore` groups configuration, weights, and history under one context and
model directory. This grouping is the persistence boundary used by model
loading and experiment restoration.

## Detailed implementation

### Resolve paths

[`GettersAndSettersProxy`](../../../src/msi_autoencoder_wrapper/core/mixins/workspace/proxies/getters_and_setters_proxy.py)
normalizes image selection and exposes dataset, catalog, model, configuration,
and latent directories. Direct image paths retain their parent directory rather
than being copied.

### Create structure

[`HelpersProxy`](../../../src/msi_autoencoder_wrapper/core/mixins/workspace/proxies/helpers_proxy.py)
creates required roots and image/model subtrees. Creation is idempotent and
restricted to the configured workspace.

### Save artifacts

[`ModelStore`](../../../src/msi_autoencoder_wrapper/core/mixins/workspace/model_store.py)
writes JSON configuration, Torch state dictionaries, and history. The stored
schema-v2 configuration includes context/cohort, dataset, model, training, and
trained-state information required for restoration.

`export_model_folder()` copies the whole artifact and refuses an existing
destination unless overwrite is explicit.
