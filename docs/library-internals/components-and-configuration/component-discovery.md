# Component discovery

Component discovery makes concrete implementations available to manager
factories without importing every implementation from the package root.

## General abstraction

### Elements of the abstraction

The discovery system separates five responsibilities:

1. A base class defines the contract of one component family.
2. An implementation module defines a concrete class.
3. A registration decorator validates that class and assigns a public key.
4. A manager owns the registry and exposes discovery and factory operations.
5. Shared utilities scan modules, validate targets, and instantiate components.

This separation allows a context or configuration to select a component by a
stable string without depending on its implementation module. It also allows
already initialized objects and compatible classes to pass through the same
manager boundary.

### Registry shapes

Most component families use one mapping:

```text
public strategy key -> implementation class
```

Readers and dataset sources follow this shape. Binning uses separate forward
and inverse registries because the two contracts differ.

Model architectures require additional ownership information:

```text
model family -> component category -> strategy key -> implementation class
model family -> component category -> required base class
```

The second mapping prevents autoencoder contracts from being applied to future
model families such as diffusion models. Each family registers the categories
and base classes that belong to that family.

### Discovery boundary

Discovery operates on importable Python packages, not arbitrary filesystem
paths. Its purpose is to execute registration decorators by importing modules.
It does not instantiate components and does not create component configuration.

Configuration and discovery meet only during resolution: configuration names a
target, while discovery must have populated the registry containing that
target.

## Detailed implementation

### Scan implementation packages

[`discover_modules()`](../../../src/msi_autoencoder_wrapper/utils/module_search.py)
accepts an import path or imported package. It resolves the package, reads its
`__path__`, and scans it recursively by default with `pkgutil.walk_packages()`.

For every discovered module it:

1. derives path segments relative to the scanned package;
2. skips segments beginning with `_`;
3. skips exact segments supplied through `excluded_parts`;
4. records the fully qualified module name;
5. imports the module unless it already exists in `sys.modules`.

The returned list contains all considered implementation modules, including
modules already imported. Repeated discovery therefore reports the same scope
while Python's module cache prevents repeated execution of module bodies.

An invalid package or failed implementation import becomes
`ProjectConfigError`. The error includes the package or module name, so a caller
does not receive a partially successful discovery result without context.

### Execute registration decorators

A concrete implementation registers at module-import time. For example,
[`PyImzMLReader`](../../../src/msi_autoencoder_wrapper/readers/strategies/pyimzml_reader.py)
is decorated with `ReaderManager.register_loader("PyImzMLReader")`.

The decorator returned by the manager:

1. receives the implementation class;
2. calls `validate_subclass()` with the family base class;
3. stores the class under the declared key;
4. returns the unchanged class so normal class import semantics are preserved.

[`validate_subclass()`](../../../src/msi_autoencoder_wrapper/utils/validators.py)
requires an actual class inheriting from the expected base. Invalid
registrations raise `IncompatibleInterfaceError` before entering the registry.

The manager registries and decorators are implemented separately for each
family, for example:

- [`ReaderManager`](../../../src/msi_autoencoder_wrapper/readers/readers_manager.py);
- [`BinnerManager`](../../../src/msi_autoencoder_wrapper/binners/binners_manager.py);
- [`DatasetSourceManager`](../../../src/msi_autoencoder_wrapper/dataset_management/sources/source_manager.py);
- [`DatasetManager`](../../../src/msi_autoencoder_wrapper/models/datasets/dataset_manager.py).

### Discover model architectures

[`ArchitecturesManager`](../../../src/msi_autoencoder_wrapper/models/architectures/architectures_manager.py)
owns model-family registries. `discover_architectures()` scans the architecture
package and excludes `schema` segments because schemas define contracts rather
than concrete discoverable implementations.

Before registering a model component, a family calls
`register_component_category(model_type, category, expected_base)`. The
resulting contract is stored in `_COMPONENT_BASES[model_type][category]`.
`register_component()` retrieves this family-owned base and validates the
implementation before storing it in
`_COMPONENT_REGISTRY[model_type][category][name]`.

Model assembly later resolves every configured component against both the
family/category registry and the corresponding family/category base class. A
missing category contract is a validation error rather than an implicit
autoencoder fallback.

### Resolve a requested component

Managers delegate factory behavior to
[`resolve_component()`](../../../src/msi_autoencoder_wrapper/utils/validators.py).
Resolution supports three target forms:

```text
registered string key -> find class -> validate constructor kwargs -> instantiate
compatible class      -> validate constructor kwargs -> instantiate
compatible instance   -> return the existing instance
```

Before these branches, `validate_component_target()` verifies that a string is
registered and that class or instance targets satisfy the expected family
contract. `validate_constructor_kwargs()` inspects required keyword-only and
positional-or-keyword constructor parameters before instantiation. Missing
parameters become `ProjectConfigError` with the target and missing names.

The context manager adds runtime dependencies such as `active_context`, file
paths, and matching forward binners before calling the resolver. These injected
objects belong to context orchestration, not discovery itself.

### Initialize discovery during runtime

[`ContextManagerProxy`](../../../src/msi_autoencoder_wrapper/core/mixins/context_manager/context_manager_mixin.py)
discovers reader and binner strategies during initialization. Dataset managers
and dataset-source workflows trigger their own discovery at the boundary where
those families are needed. Architecture discovery is triggered through the
model system.

Annotation readers are the current explicit-loading exception.
`AnnotationReaderManager.load_builtin_readers()` imports the SQLite and
METASPACE CSV modules directly. Their decorators still populate a registry, and
all later resolution uses the same `resolve_component()` path.

### Preserve discovery invariants

The mechanism depends on these invariants:

- implementation modules must be importable without creating a configured
  runtime object;
- registration keys must identify one implementation within their registry;
- decorators must validate the correct family-owned base contract;
- schema and abstract-only packages must not be treated as implementations;
- discovery must complete before resolving a string key;
- constructor validation must occur before instantiating a selected class.

Instructions for adding or testing implementations belong in the
[developer guides](../../developer-guides/index.md). Public component selection
belongs in the corresponding [how-to guide](../../how-to/index.md).
