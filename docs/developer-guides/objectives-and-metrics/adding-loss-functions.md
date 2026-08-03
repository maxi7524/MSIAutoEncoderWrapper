# Add a loss function

A loss function selects model and batch tensors, invokes an optimization
measure, and returns a scalar compatible with the composite loss.

## Criterion scope and contracts

### Criterion families

Autoencoder criteria derive from reconstruction, contrastive, or head bases.
The base determines tensor selection and lifecycle expectations.

### Lifecycle hooks

`on_phase_start()` prepares dataset/model-dependent state once per phase.
`on_batch_start()` may add or transform batch views. `forward()` computes loss
without mutating persistent batch data.

## Implementation instructions

### Implement and register

Inherit the narrowest criterion base and register under model type and criterion
group through `CriterionsManager`. Constructor parameters must be portable.

For head criteria, use the named-head output and target binding supplied by the
composite-loss builder. Do not hard-code one global target key.

### Reuse metrics

Put reusable numerical distance or score logic in `metrics`. The criterion
selects tensors and applies reduction/weighting. This lets analysis compute the
same measure without importing training code.

### Test numerical validity

Test exact small examples, reduction, masks, missing outputs, gradients, finite
results, and domain constraints. Non-negative spectral objectives must reject
negative inputs rather than produce a meaningless scalar.
