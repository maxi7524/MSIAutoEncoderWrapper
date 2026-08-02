# Implement configurable components

Every configurable implementation must use the shared configuration contract so
component managers and saved experiments can reconstruct it consistently.

## Configuration scope and invariants

### Portable state

Configuration contains constructor state required to reproduce behavior. It
must exclude live contexts, open handles, caches, tensors that are not scalar,
credentials, and process-local objects.

### Shared interface

`ConfigurableComponent` provides `get_config()`, `export_config()`, and
`from_config()`. Override only when constructor parameters require explicit
normalization or runtime dependency handling.

## Implementation instructions

### Store constructor state

```python
from msi_autoencoder_wrapper.configuration import ConfigurableComponent


class ExampleComponent(ConfigurableComponent):
    def __init__(self, width: int, active_context=None) -> None:
        self.width = int(width)
        self.active_context = active_context
        self._config = {"width": self.width}
```

Use English keys and JSON-compatible values. Paths may remain `Path` values;
the shared exporter converts them to strings.

### Handle runtime dependencies

The default `from_config(parameters, **dependencies)` calls the constructor with
both mappings. Managers inject `active_context`, `cohort_context`, devices, or
other runtime owners at restoration time.

### Add round-trip tests

Export the component, reconstruct it through the family manager or
`from_config()`, and compare behaviorally relevant configuration. Include the
new implementation in registry-wide round-trip tests so future components
cannot bypass the contract.
