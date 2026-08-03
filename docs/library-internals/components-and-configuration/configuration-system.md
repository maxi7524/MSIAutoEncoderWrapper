# Configuration system

The configuration system gives every configurable component one recursive,
JSON-compatible representation and coordinates complete experiment restoration.

## General abstraction

### Component-owned parameters

Each component owns the constructor parameters required to rebuild itself.
Managers and the configuration orchestrator do not duplicate component-specific
schemas.

### Runtime dependencies

Portable configuration excludes live wrapper, context, device cache, and open
reader objects. `from_config()` receives these runtime dependencies explicitly.

## Detailed implementation

### Export components

[`ConfigurableComponent`](../../../src/msi_autoencoder_wrapper/configuration/components.py)
defines `get_config()`, `export_config()`, and `from_config()`. Export nodes
contain `type`, optional `module`, `version`, and JSON-compatible `parameters`.

`make_json_compatible()` handles primitive values, paths, mappings, sequences,
classes, scalar tensors, and nested configurable components. Unsupported values
raise `ProjectConfigError` with their configuration path.

### Assemble consolidated schema

Workspace model saving combines context or cohort configuration, dataset
descriptor, model graph, training configuration, and artifact state into root
schema version 2. [`schema.py`](../../../src/msi_autoencoder_wrapper/configuration/schema.py)
requires `experiment`, `data`, and `model` sections.

### Restore runtime state

[`ConfigurationOrchestrator`](../../../src/msi_autoencoder_wrapper/configuration/orchestrator.py)
delegates context restoration to context/cohort managers, dataset restoration
to `DatasetManager`, and model restoration to `ModelLoader`. It coordinates
order but does not reconstruct component internals itself.

### Maintain round-trip invariants

For each registered configurable component, exporting parameters and calling
its reconstruction path must preserve behaviorally relevant constructor state.
Registry-wide tests enforce this invariant for new implementations.
