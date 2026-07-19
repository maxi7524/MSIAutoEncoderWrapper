# Custom local models and components

This guide describes the currently supported local-model extension path. A local
model is associated with one image context. Support for model families trained
across multiple independent images is planned but is not part of the current API.

## Runtime contexts

The wrapper deliberately distinguishes two model references:

- `wrapper.models_manager.loaded_model` is the one PyTorch model currently loaded
  for configuration, training, or inference.
- `wrapper.models_manager.model_functionality` is the registered high-level
  interface for that loaded model. For an autoencoder it exposes `encode`,
  `decode`, `transform`, and `compress_to_file`.
- `wrapper.active_context.local_model_functionality` is the interface preserved
  in the selected image ledger.
- `wrapper.active_context.model_functionality` prefers the local interface and
  falls back to the currently loaded interface when no local model is bound.

This separation allows a trained autoencoder to remain usable for one image while
another model is loaded in `models_manager`:

```python
wrapper.models_manager.attach_model(
    local_autoencoder,
    model_name="bladder-autoencoder",
    trained=True,
    bind_to_local_context=True,
)

local_functionality = wrapper.active_context.model_functionality

wrapper.models_manager.load_model(
    "global",
    "another-model",
    bind_to_local_context=False,
)

assert wrapper.active_context.model_functionality is local_functionality
```

Use `wrapper.models_manager.bind_loaded_model_to_local_context()` when a model
was loaded first and should be attached to the currently configured image later.

## Architecture families

`ArchitecturesManager` registers master model families. The library currently
ships one complete family, `autoencoder`. Other local families can be added, but
they require both an architecture contract and a matching runtime-functionality
adapter before they can expose user-facing operations.

> **TODO — global and multi-image models:** document the dedicated contracts once
> datasets spanning multiple image contexts and their training lifecycle exist.

## Adding autoencoder components

An autoencoder is assembled from independently registered components:

- an `encoder` producing `latent_space`;
- an optional `decoder` producing `reconstruction`;
- an optional `projector` producing `projection` for contrastive objectives;
- optional named `heads`, exposed as `head_<name>` outputs.

Each component must:

1. inherit from the relevant base component or a compatible `torch.nn.Module`;
2. save constructor parameters in `_config`;
3. use an English, Sphinx-compatible docstring;
4. be registered under the correct model family and component category.

Example encoder:

```python
import torch
import torch.nn as nn

from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)
from msi_autoencoder_wrapper.models.architectures.types.autoencoders.encoders.base_encoder import (
    MSIBaseEncoder,
)


@ArchitecturesManager.register_component("autoencoder", "encoder", "LinearEncoder")
class LinearEncoder(MSIBaseEncoder):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self._config = {"input_dim": input_dim, "latent_dim": latent_dim}
        self.network = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

The registered name can then be used in portable configuration:

```python
wrapper.models_manager.set_model_type("autoencoder", "linear-autoencoder")
wrapper.models_manager.set_component(
    "encoder",
    "LinearEncoder",
    input_dim=1500,
    latent_dim=32,
)
```

Ready component instances and component classes are also accepted, but registered
names plus JSON-compatible parameters are preferred because they can be recreated
from saved configuration.

## Custom presets

A preset is a function returning component strategies and parameters. It may use
the active reader and binner to derive dimensions:

```python
from msi_autoencoder_wrapper.models.architectures.architectures_manager import (
    ArchitecturesManager,
)


@ArchitecturesManager.register_preset("autoencoder", "SmallLinear")
def small_linear_preset(active_context, latent_dim: int = 16):
    input_dim = active_context.binner.GetXAxisDepth()
    return {
        "encoder": {
            "strategy": "LinearEncoder",
            "params": {"input_dim": input_dim, "latent_dim": latent_dim},
        },
    }
```

Apply it after selecting the model family and configuring an image context:

```python
wrapper.models_manager.set_model_type("autoencoder", "small-linear")
wrapper.models_manager.set_model_preset("SmallLinear", latent_dim=16)
```

## Custom criteria

Criteria are registered by model family and execution category. Reconstruction,
contrastive, and head objectives have different input contracts and must inherit
from their matching base class. See [CRITERIONS.md](CRITERIONS.md) for the registry,
training configuration, lifecycle hooks, and Masserstein details.

## Configuration requirements

All reconstructable components must return JSON-compatible constructor state from
`get_config()`/`GetConfig()`. Do not place runtime objects, datasets, readers,
open file handles, devices, or tensor data in `_config`. Store only values needed
to recreate the component.
