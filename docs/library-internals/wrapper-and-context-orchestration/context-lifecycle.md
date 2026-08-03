# Context lifecycle

An image context moves from workspace identity through component registration to
lazy runtime activation and optional local model binding.

## General abstraction

### Local state

The context ledger is keyed by image name. Each bucket contains reader,
annotation reader, binner, inverse binner, normalization, local model
functionality, and transient data.

### Active routing

The active-context proxy caches one ledger bucket. Workspace activation changes
the selected key; the proxy re-resolves when its instantiated key no longer
matches.

## Detailed implementation

### Register components

[`ContextManagerProxy._set_component()`](../../../src/msi_autoencoder_wrapper/core/mixins/context_manager/context_manager_mixin.py)
is wrapped by `manage_image_context`. The decorator activates the requested
image temporarily, allowing setters to use one injection path.

The setter selects a registry and expected base, injects file path,
`active_context`, or forward binner when required, resolves the target, and
stores the resulting instance in the image bucket.

### Resolve active state

[`ActiveContextProxy`](../../../src/msi_autoencoder_wrapper/core/mixins/active_context/active_context_mixin.py)
reads the active image key, falls back to the workspace default, then copies
bucket references into local caches. Missing image or missing ledger entry is a
validation error.

### Bind model functionality

Local model functionality is stored inside the image bucket. The active context
prefers this local interface over the separately loaded model-manager runtime.
This preserves image-local inference when another model is loaded for
comparison or cohort work.

### Serialize and restore

`get_context_config()` exports component descriptors. `load_context_config()`
resolves relative paths against an optional base, reconstructs components
through their managers, and restores normalization in the selected image
bucket.
