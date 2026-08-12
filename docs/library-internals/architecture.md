# Library architecture

The library separates persistent workspace state, local and cohort contexts,
data transformation, model construction, training, execution, analysis, and
external dataset management.

## General abstraction

### Facade and subsystem ownership

`MSIAutoEncoderWrapper` is a cooperative-mixin facade. It exposes subsystem
proxies but does not implement readers, models, metrics, or provider adapters.

```text
MSIAutoEncoderWrapper
├── workspace                 filesystem identity and artifacts
├── context_manager           per-image component ledger
├── active_context            lazy routing to one local context
├── cohorts                   immutable multi-image context values
└── models_manager            dataset, architecture, runtime, training
```

Independent top-level packages own data contracts, component implementations,
dataset management, experiment execution, metrics, visualization, and analysis.

### Dependency direction

Low-level data records and configuration contracts do not depend on the wrapper.
Managers depend on base contracts and shared factories. Context proxies inject
runtime dependencies. Training and analysis consume configured contexts and
models. Provider adapters terminate at the canonical local catalog.

## Detailed implementation

### Core orchestration

[`core/wrapper.py`](../../src/msi_autoencoder_wrapper/core/wrapper.py) combines
workspace, context, cohort, active-context, spatial, and model-manager mixins.
Each mixin creates a proxy during cooperative initialization.

[`core/mixins/context_manager`](../../src/msi_autoencoder_wrapper/core/mixins/context_manager/__init__.py)
owns the per-image ledger. [`active_context`](../../src/msi_autoencoder_wrapper/core/mixins/active_context/__init__.py)
resolves that ledger lazily. [`cohort`](../../src/msi_autoencoder_wrapper/core/mixins/cohort/__init__.py)
captures configured local contexts as immutable members.

### Computational pipeline

[`data`](../../src/msi_autoencoder_wrapper/data/__init__.py) defines raw, dense, latent,
inverse, target, and spatial contracts. [`readers`](../../src/msi_autoencoder_wrapper/readers/__init__.py),
[`binners`](../../src/msi_autoencoder_wrapper/binners/__init__.py), and
[`normalization`](../../src/msi_autoencoder_wrapper/normalization/__init__.py) transform
these representations before model execution.

[`models`](../../src/msi_autoencoder_wrapper/models/__init__.py) owns datasets and
architecture families. [`training`](../../src/msi_autoencoder_wrapper/training/__init__.py)
owns criteria and the phase engine. [`metrics`](../../src/msi_autoencoder_wrapper/metrics/__init__.py)
owns numerical measures independent of presentation.

### Persistence and external boundaries

[`configuration`](../../src/msi_autoencoder_wrapper/configuration/__init__.py) defines
component and consolidated configuration contracts. Workspace `ModelStore`
writes model artifacts. [`execution`](../../src/msi_autoencoder_wrapper/execution/__init__.py)
turns schema-v1 campaign YAML into tasks for local or Slurm execution.

[`msi_dataset_manager`](../../packages/msi_dataset_manager/src/msi_dataset_manager/__init__.py)
is an independent distribution (`packages/msi_dataset_manager`), not a module
of this wrapper. It isolates provider communication and normalizes external
records into SQLite and local imzML artifacts. Annotation readers consume
those local forms without calling providers; see
[Dataset management internals](dataset-management/index.md).

### Analysis boundary

[`analysis`](../../src/msi_autoencoder_wrapper/analysis/__init__.py) is currently
provisional. Its target design is orchestration over `metrics` and
`visualization`, not duplication of numerical or plotting implementations.
