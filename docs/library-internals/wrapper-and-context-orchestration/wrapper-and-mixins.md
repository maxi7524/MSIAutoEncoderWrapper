# Wrapper and mixins

The wrapper composes subsystem mixins into one facade while each subsystem keeps
its state in a dedicated proxy.

## General abstraction

### Composition model

The facade uses cooperative `super()` initialization. Mixins do not subclass one
another by domain; Python's method-resolution order initializes them in the
order declared by `MSIAutoEncoderWrapper`.

### Proxy boundary

Public properties such as `workspace`, `context_manager`, `cohorts`,
`active_context`, and `models_manager` point to proxies holding domain state.
This prevents the facade object from becoming the storage location for every
registry and cache.

## Detailed implementation

### Initialize the facade

[`MSIAutoEncoderWrapper`](../../../src/msi_autoencoder_wrapper/core/wrapper.py)
selects a default device, stores it before cooperative initialization, and
passes workspace layout and coordinate order through the MRO.

The mixin order is spatial context, workspace, context manager, cohort, active
context, and models manager. Earlier mixins may therefore rely on attributes
created explicitly by the wrapper but must forward unknown arguments.

### Share wrapper state safely

Proxies derive from `BaseWrapperProxy` or retain a wrapper reference. They read
shared identity through narrow properties such as project path and active image
key. Runtime components receive `active_context` through dependency injection
rather than importing the facade.

### Preserve boundaries

Workspace resolves paths; context manager stores per-image components; active
context routes access; model manager owns the loaded model and dataset. Moving
state between these owners creates ambiguous persistence and activation rules.
