# Add provider filters

Provider filters expose query capabilities and derived local statistics without
changing the shared explorer workflow.

## Filter scope and semantics

### Native and derived filters

Native filters are sent to the provider. Derived filters run on retrieved
metadata or annotation statistics. Document which stage owns each filter.

### Diagnostics

Rejected records must retain a reason and relevant observed value. A filter must
not silently remove records without diagnostics.

## Implementation instructions

### Define metadata and values

Add filter keys to `get_available_filters()` and enumerated values to
`get_available_values()` when finite enumeration is available. Validate types,
ranges, and incompatible combinations before network calls where possible.

### Apply filters consistently

Ensure Python explorer and CLI query use the same adapter method. Preserve the
effective filter mapping in exported selection JSON.

### Test boundaries

Test each operator at boundary values, missing provider fields, exclusions,
accepted/rejected lists, serialization, and interaction with annotation FDR.
