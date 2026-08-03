# Run and organize tests

This guide defines how a library change should be covered and validated before
review.

## Testing scope and available mechanisms

### What the test suite verifies

Tests cover isolated component contracts, registry-wide invariants, context
integration, data pipelines, model execution, and workspace persistence. The
required scope depends on the boundary crossed by the implementation change.

A local component change starts with focused tests. A change to shared
configuration, discovery, contexts, datasets, or model runtime also requires
adjacent integration tests. A full-suite run is required when the affected
boundaries cannot be enumerated confidently.

### Available fixtures and test locations

Tests are grouped by implementation domain under `tests/`. Shared fixtures in
`tests/conftest.py` provide:

- `msi_fixture_path`: the compact imzML test image;
- `mock_reader`: a lightweight reader configured for that image;
- `mock_active_context`: the reader plus a regular-grid binner.

Temporary catalogs, workspaces, configurations, and generated artifacts belong
under pytest's `tmp_path`. Tests must not mutate `data/tutorial_workspace` or
depend on state left by another test.

## Implementation instructions

### Add tests for a concrete change

#### Select the responsible test module

Place the test in the directory matching the changed component family. Extend
an existing module when it already owns the tested contract. Create a new module
only when the behavior belongs to a distinct responsibility.

Examples:

- annotation-reader behavior belongs in
  `tests/annotations/test_annotation_readers.py`;
- dataset-source orchestration belongs in `tests/dataset_sources/`;
- workspace catalog behavior belongs in
  `tests/workspace/test_dataset_catalog.py`;
- architecture registration and construction belong in `tests/architectures/`.

#### Cover success, rejection, and integration

A concrete implementation should normally test:

1. the smallest valid input;
2. its result structure and domain invariants;
3. invalid values and the expected project exception;
4. integration with the manager or context that owns it;
5. configuration round-trip when it implements the configurable-component
   contract.

Assert observable state or returned values. Do not assert log wording unless
the log is the intended user-visible result of a recoverable condition.

### Add a registered implementation

#### Verify discovery and resolution

A registered component requires a test proving that discovery or explicit
built-in loading exposes its public key. Resolve that key through the manager;
importing the concrete class directly does not verify registry integration.

Also test rejection of a class that violates the family base contract when the
registration API is part of the change.

#### Reuse registry-wide contract tests

Use pytest parametrization when all registered implementations must satisfy the
same rule. Iterate the registry after discovery and run the shared assertion for
every entry. This makes a future implementation enter the contract test without
adding its name manually.

Registry-wide checks are appropriate for:

- configuration export and `from_config()` round-trip;
- required output type and shape;
- finite and domain-valid numerical outputs;
- required public methods;
- constructor or metadata invariants shared by the family.

Implementation-specific behavior remains in a separate focused test; it should
not be forced into a family-wide assertion.

### Run focused validation

#### Use the project interpreter

Run pytest from the repository root with the virtual-environment interpreter:

```bash
.venv/bin/python -m pytest -q tests/annotations/test_annotation_readers.py
```

`python -m pytest` ensures that pytest and the package use the same interpreter.
The project declares Python 3.10 or newer. A system `pytest` executable may
belong to another Python installation and must not be used as evidence of a
project test result.

#### Include adjacent boundaries

For an annotation change that also affects import and catalog resolution, run:

```bash
.venv/bin/python -m pytest -q \
  tests/annotations/test_annotation_readers.py \
  tests/dataset_sources/test_pipeline.py \
  tests/workspace/test_dataset_catalog.py
```

Choose adjacent modules from the actual data or object flow. Do not add unrelated
test directories only to increase the number of executed tests.

### Run repository-level validation

#### Execute the complete suite

Run all tests when the change affects shared infrastructure or after focused
tests pass:

```bash
.venv/bin/python -m pytest -q
```

#### Execute configured static checks

Run formatter, linter, and type-checker commands only when those tools are
configured and installed for the repository. Do not report an unavailable tool
as a passed check. `git diff --check` can independently detect whitespace errors:

```bash
git diff --check
```

Review the final diff for unrelated edits, temporary diagnostics, generated
files, and accidental changes to user-owned artifacts.

### Report the validation result

The review handoff must state:

- exact commands that executed;
- number of passed, failed, skipped, or deselected tests;
- formatter, linter, type-checker, and documentation-build results;
- checks that were not executed and the reason;
- remaining assumptions or untested integration boundaries.

A command that fails during collection, uses the wrong interpreter, or cannot
find its requested test file did not validate the implementation.
