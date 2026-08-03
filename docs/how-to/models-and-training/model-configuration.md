# Configure a model

The model manager builds one registered architecture family from registered
components and binds it to the configured dataset.

## Purpose and available operations

### Configuration paths

A model can be assembled component by component, populated by a registered
preset, attached as an initialized Torch module, or restored from a saved model
artifact.

### Discovery operations

The manager can list model families, family-owned component categories,
implementations in a category, presets, datasets, and loss functions.

## Detailed instructions

### Inspect registered options

```python
wrapper.models_manager.get_available_model_types()
wrapper.models_manager.set_model_type("autoencoder", "baseline")
wrapper.models_manager.get_available_component_categories()
wrapper.models_manager.get_available_components("encoder")
wrapper.models_manager.get_available_model_presets()
wrapper.models_manager.get_available_datasets()
```

`print_return` controls formatted output and `return_value` returns structured
reflection data for each listing method.

### Configure a family and components

```python
wrapper.models_manager.set_model_type(
    model_type="autoencoder",
    model_name="baseline",
)
wrapper.models_manager.set_component(
    category="encoder",
    name="CNNEncoder",
    input_dim=1000,
    latent_dim=16,
    channels=[1, 8],
    kernels=[3],
    strides=[2],
    spatial_dims=[1000, 499],
)
```

`set_component()` accepts a registry key, compatible class, or initialized
component. Constructor parameters depend on the selected implementation.
Categories are validated against the active model family.

### Apply a preset

```python
wrapper.models_manager.set_model_type("autoencoder", "baseline")
wrapper.models_manager.set_model_preset(
    "GradualReduction",
    latent_dim=16,
    projection_dim=128,
    user_hyperparameters={},
)
```

A preset fills the building buffer; it does not compile or train the model.
Required preset parameters and available overrides are returned by
`get_available_model_presets()`.

### Compile and inspect configuration

```python
model = wrapper.models_manager.compile_model(run_validation_pass=True)
config = wrapper.models_manager.get_model_config()
wrapper.models_manager.print_model_config()
```

Compilation requires a dataset and all components required by the selected
family. The optional validation pass performs a forward check using the active
dataset contract.
