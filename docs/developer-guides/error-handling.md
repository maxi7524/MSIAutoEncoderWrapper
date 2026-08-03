# Implement error handling

Validation errors must stop invalid configuration or numerical state at the
boundary where it becomes identifiable.

## Error scope and categories

### Project exceptions

Use `ValidationError` for invalid values or state, `ProjectConfigError` for
unresolvable configuration, `IncompatibleInterfaceError` for contract mismatch,
and model-specific exceptions for architecture construction failures.

### Recoverable absence

Optional state may return `None` with a warning only when the public operation
defines absence as valid. Automatic annotation detection is one example; an
explicit missing catalog is an error.

## Implementation instructions

### Validate at boundaries

Check paths before I/O, constructor arguments before instantiation, tensor
shape/domain before computation, and persisted schema before restoration. Error
messages must name the component and invalid field or state.

### Preserve exception context

Translate exceptions only when adding domain context or changing abstraction.
Use `logger.error(..., exc_info=True)` when a caught exception is contextualized
or re-raised. Do not add `try/except` blocks only for logging.

### Test rejection behavior

Assert the project exception type and a stable semantic fragment. Include
negative, non-finite, missing, incompatible, and ambiguous inputs where the
domain defines them as invalid.
