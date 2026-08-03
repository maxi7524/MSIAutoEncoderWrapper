# Add projectors, heads, and presets

Projectors prepare latent embeddings for auxiliary objectives, heads predict
dataset targets, and presets generate compatible component configurations.

## Component scope and invariants

### Projectors and heads

Projectors inherit `MSIBaseProjector`. Heads inherit `MSIBaseHead` and return
logits compatible with their target type and criterion.

### Presets

Presets calculate configuration only. They may inspect active context dimensions
but must not instantiate, train, or save a model.

## Implementation instructions

### Register components

Use `register_component("autoencoder", "projector", name)` or category `head`.
Head output dimension must equal the dataset target schema class count.

### Register a preset

Decorate a callable with `register_preset("autoencoder", name)`. Return model,
component, and dataset buffer values expected by `set_model_preset()`. Treat user
overrides explicitly and validate derived dimensions.

### Test target binding

Compile at least two named heads, bind them to target fields, run forward, save,
reload, and verify both logits and `head_specs`.
