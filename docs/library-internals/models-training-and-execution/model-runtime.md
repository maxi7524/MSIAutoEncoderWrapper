# Model runtime

Runtime state separates an arbitrary loaded model from model functionality bound
to one image context.

## General abstraction

### Loaded model

`ModelsManager` owns the current Torch module, model type, model name, dataset,
and model-family runtime interface.

### Local model

An image ledger may retain its own functionality interface. Active-context
resolution prefers that local interface, allowing another model to be loaded
without muting the image-local model.

## Detailed implementation

### Attach and detect

[`ModelRuntimeProxy.attach_model()`](../../../src/msi_autoencoder_wrapper/core/mixins/models_manager/proxies/model_runtime_proxy.py)
accepts a Torch module, detects known model families when type is omitted,
creates the family runtime interface, records trained state, and optionally
binds it locally.

### Enforce trained state

[`AutoencoderContextInterface`](../../../src/msi_autoencoder_wrapper/core/mixins/active_context/autoencoder_context_manager.py)
guards encode, decode, transform, and latent export. Compilation attaches an
untrained interface; successful training or artifact loading marks it trained.

### Load artifacts

[`ModelLoader`](../../../src/msi_autoencoder_wrapper/models/model_loader.py)
resolves an artifact directory, validates its configuration, rebuilds the graph,
loads weights with optional strictness, and calculates folder fingerprints used
by cohort references.

### Unload without deleting

Unloading clears current model-manager runtime and transient training state. It
does not delete workspace artifacts or an independently bound local model.
