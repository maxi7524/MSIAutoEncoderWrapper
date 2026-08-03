# Maintain public exports

Public package exports provide short imports for frequently constructed user
components while implementation modules remain organized by responsibility.

## Export scope and constraints

### Appropriate exports

Facade classes, base contracts intended for extension, manager factories, and
strategies commonly instantiated manually are candidates for the nearest
package `__init__.py`.

### Internal symbols

Private helpers, caches, validators used only by one subsystem, and concrete
implementation details should remain module imports.

## Implementation instructions

### Export at the nearest stable level

Add the import and its name to `__all__` in the responsible package. For example,
annotation strategies are available from `msi_autoencoder_wrapper.annotations`
while the facade is available from the package root.

### Avoid compatibility aliases

Do not add deprecated aliases or migration import paths. Compatibility is reset
for the current restructuring. Retain intentionally capitalized reader methods
required by the m2aia API contract.

### Test the import

Add one import test for the documented path and retain implementation tests in
the component's domain. Public export tests should not duplicate behavior tests.
