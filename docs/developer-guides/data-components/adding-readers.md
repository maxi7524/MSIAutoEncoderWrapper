# Add an MSI reader

A reader translates one storage backend into the shared spectrum, coordinate,
metadata, and batch contract.

## Reader scope and invariants

### Required behavior

Inherit `MSIBaseReader` and implement spectrum count, spectrum values, spectrum
position, mass bounds/axis when available, and metadata. Preserve the
capitalized API required for m2aia compatibility.

### Batch behavior

Implement `GetSpectrumBatch()` when the backend can read several spectra more
efficiently. Declare whether the batch uses one shared mass axis.

## Implementation instructions

### Implement and register

Place the module under `readers/strategies` and decorate it with
`ReaderManager.register_loader(name)`. Store file path and portable constructor
options in configuration.

### Validate output

Mass and intensity arrays must be one-dimensional, equally sized, and finite.
Positions must be stable for each zero-based spectrum ID. Batch sample order
must match requested IDs.

### Add tests

Test discovery, configuration, single/batch equivalence, shared and variable
axes, invalid paths, coordinate lookup, and interaction with `PixelDataset`.
