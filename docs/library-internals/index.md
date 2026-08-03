# Library internals

These documents explain how the library coordinates components, data, state,
and execution. They describe existing behavior rather than modification steps.

## Contents

- [Library architecture](architecture.md) — top-level module responsibilities and dependency directions.
- [Wrapper and context orchestration](wrapper-and-context-orchestration/index.md) — facade composition, workspace state, local contexts, and cohort execution.
- [Components and configuration](components-and-configuration/index.md) — component discovery, registries, configuration serialization, and reconstruction.
- [Models, training, and execution](models-training-and-execution/index.md) — architecture assembly, runtime binding, training phases, and experiment campaigns.
- [Metrics, analysis, and visualization](metrics-analysis-and-visualization/index.md) — numerical, analytical, and rendering responsibility boundaries.
- [Data pipeline](data-pipeline/index.md) — representations, batch movement, preprocessing, targets, and reconstruction.
- [Dataset management internals](dataset-management/index.md) — provider boundaries, artifacts, annotation normalization, SQLite, and merge provenance.

```{toctree}
:hidden:

architecture
wrapper-and-context-orchestration/index
components-and-configuration/index
models-training-and-execution/index
metrics-analysis-and-visualization/index
data-pipeline/index
dataset-management/index
```
