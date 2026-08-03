# Rewrite and extend analysis

Analysis extensions must contain analytical selection and aggregation logic,
while reusable calculations and plots remain in global modules.

## Analysis scope and target structure

### Responsibility boundary

`metrics` owns numerical measures. `visualization` owns reusable plotting and
themes. `analysis` prepares data for a question, invokes metrics, compares or
aggregates results, and composes returned analytical records.

### Current status

The current autoencoder analysis implementation duplicates some metric and view
logic. Treat it as migration input, not the template for new stable domains.

## Implementation instructions

### Define one analytical question

Specify required retained arrays, selection rules, metrics, aggregation, result
schema, and optional views. Add memory estimation before materializing full
dataset results.

### Extract shared operations first

Move reusable formulas to `metrics` and generic rendering to `visualization`.
Analysis may wrap them but must not copy their implementations.

### Integrate single and multi-model paths

Use shared prepared-result contracts and explicit model names. Preserve source
sample IDs and spatial mapping. Compare models on identical selected samples and
representation spaces.

### Add tests

Test preparation retention, memory estimates, metric delegation, spatial
mapping, target masks, multi-model alignment, result schemas, plotting return
types, and absence of duplicated numerical implementations.
