# Model architecture system

The architecture system assembles registered model-family graphs from
family-owned component categories and portable parameter dictionaries.

## General abstraction

### Family ownership

A model family owns its master architecture, component-category contracts,
component implementations, and presets. Autoencoder encoder/decoder contracts
are not global architecture contracts.

### Buffered construction

`ArchitectureProxy` stores user selections in a building buffer. Compilation
turns that buffer into component instances, then injects them into the master
architecture constructor.

## Detailed implementation

### Register contracts and implementations

[`ArchitecturesManager`](../../../src/msi_autoencoder_wrapper/models/architectures/architectures_manager.py)
stores `_MODEL_REGISTRY`, `_COMPONENT_REGISTRY`, `_COMPONENT_BASES`, and
`_PRESET_REGISTRY`. A family registers each category base before implementation
decorators use it.

### Buffer public selections

[`ArchitectureProxy`](../../../src/msi_autoencoder_wrapper/core/mixins/models_manager/proxies/architecture_proxy.py)
validates model type, category, and target before retaining the target and
constructor kwargs. Named heads are stored as a nested collection with separate
target-field bindings.

### Assemble the graph

`build_model()` resolves direct components and nested heads against their
family/category registry. It then resolves the master architecture with the
`resolved_components` mapping. Missing contracts and incompatible classes fail
before a graph is returned.

### Compile with a dataset

The proxy constructs the selected dataset after graph assembly, attaches both
to wrapper runtime state, creates the model-family functionality interface, and
optionally validates one forward pass.
