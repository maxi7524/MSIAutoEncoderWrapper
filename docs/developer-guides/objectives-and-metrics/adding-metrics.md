# Add a metric

A metric computes a reusable numerical result for a declared object space and
representation domain.

## Metric scope and contracts

### Object spaces

Metrics are grouped by spectra, classification decisions, classes, and
embeddings. The group determines input structure and aggregation semantics.

### Compatibility requirements

A metric may require non-negative values, linear intensity, sample-wise scalar
normalization compatibility, a mass axis, or a specific output space.

## Implementation instructions

### Implement the numerical function

Keep plotting, dataset iteration, and model execution outside the metric. Return
per-sample values or sufficient statistics when dataset-level aggregation is
required.

### Register requirements

Add the metric to the global registry with explicit requirements. Compatibility
checks must run before expensive computation. Masserstein-like metrics may be
computationally expensive but remain metrics; provide batching/device options
rather than moving them into analysis.

### Integrate training and analysis

Criteria call the metric and reduce for optimization. Analysis retains or
aggregates metric output and passes prepared values to visualization.

### Test the metric

Test known values, symmetry/order properties where applicable, finite output,
shape, device consistency, invalid domains, normalization compatibility, and
registry-wide invocation.
