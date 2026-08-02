# Configure autoencoders

The autoencoder family combines an encoder, decoder, optional projector, and
zero or more named heads.

## Purpose and available operations

### Autoencoder outputs

The model returns reconstruction and latent representations. A variational
encoder may add distribution parameters. Each named head adds an output bound
to one dataset target field.

### Non-negative reconstruction

Decoder output activation controls the reconstructed intensity domain. Use a
non-negative output activation for MSI intensity reconstruction.

## Detailed instructions

### Configure encoder and decoder

```python
wrapper.models_manager.set_model_type("autoencoder", "example-ae")
wrapper.models_manager.set_component(
    "encoder",
    "CNNEncoder",
    input_dim=1000,
    latent_dim=16,
    channels=[1, 8],
    kernels=[3],
    strides=[2],
    spatial_dims=[1000, 499],
)
wrapper.models_manager.set_component(
    "decoder",
    "CNNDecoder",
    latent_dim=16,
    channels=[1, 8],
    kernels=[3],
    strides=[2],
    spatial_dims=[1000, 499],
    output_activation={"type": "softplus", "parameters": {}},
)
```

Output activation accepts the types implemented by
`build_output_activation()`. `softplus` preserves non-negativity while retaining
a smooth gradient. An identity activation is invalid for workflows requiring
non-negative reconstructed intensities.

### Configure a projector and heads

```python
wrapper.models_manager.set_component(
    "projector",
    "LinearProjector",
    input_dim=16,
    output_dim=8,
)
wrapper.models_manager.set_head(
    head_id="molecule_primary",
    target_field="molecule",
    strategy="LinearClassificationHead",
    latent_dim=16,
    output_dim=22,
)
```

`head_id` must be non-empty and cannot contain dots. `target_field` must match a
dataset target. Multiple heads may bind to the same target field.

### Use the runtime interface

After training or loading a trained model:

```python
runtime = wrapper.models_manager.autoencoder
latent = runtime.encode(spectrum)
reconstruction = runtime.decode(
    latent,
    grid_xs=True,
    output_space="source",
)
```

Untrained models reject inference through this interface. `grid_xs=True`
returns reconstruction on the configured binner axis. `output_space` controls
normalized or source-scale output when a normalization trace is available.
