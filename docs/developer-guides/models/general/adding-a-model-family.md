# Add a model family

A model family owns one master graph, its component categories, and the base
contract required by each category.

## Family scope and required elements

### Required registrations

Register the master architecture with `register_model_type()`. Register every
category through `register_component_category()` before importing decorated
implementations. Register presets only when they produce a complete family
configuration.

### Family isolation

Do not reuse autoencoder base classes as global component bases. Shared behavior
belongs in a neutral base; family-specific forward/output contracts remain in
the family package.

## Implementation instructions

### Create the package and contracts

Place the family under `models/architectures/types/<family>/`. Define a master
architecture inheriting `MSIBaseMasterArchitecture` or `nn.Module`, plus one base
class per component category.

```python
ArchitecturesManager.register_component_category(
    "diffusion", "denoiser", MSIBaseDenoiser
)
```

### Register the graph and implementations

```python
@ArchitecturesManager.register_model_type("diffusion")
class MSIDiffusionArchitecture(MSIBaseMasterArchitecture):
    ...


@ArchitecturesManager.register_component("diffusion", "denoiser", "UNet1D")
class UNet1D(MSIBaseDenoiser):
    ...
```

The master constructor must accept `resolved_components` in the form assembled
by `ArchitecturesManager.build_model()` or expose a family-specific assembly
adapter.

### Integrate the proxy

Ensure categories are discoverable and that `ArchitectureProxy.compile_model()`
can determine required dataset and runtime functionality. Do not add category
names to autoencoder-only loops; generalize the family assembly boundary when
necessary.
