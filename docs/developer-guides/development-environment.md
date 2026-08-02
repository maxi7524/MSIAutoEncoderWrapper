# Configure the development environment

Development uses the repository virtual environment and an editable source
layout under `src/`.

## Development scope and available tools

### Project requirements

The project declares Python 3.10 or newer and dependency groups through
`pyproject.toml` and `uv.lock`. CPU, CUDA 11.8, and MPS extras select Torch and
optional m2aia support.

### Configured checks

Pytest is the configured test framework. The repository currently does not
declare Ruff, Black, or a static type checker in `pyproject.toml`; do not report
those checks unless configuration is added.

## Implementation instructions

### Install an editable environment

```bash
uv sync --extra cpu --group dev
```

Select `cu118` or `mps` instead of `cpu` for the target platform. Do not silently
change pinned Torch, metaspace2020, or pridepy versions.

### Verify imports and tests

```bash
.venv/bin/python -c "import msi_autoencoder_wrapper"
.venv/bin/python -m pytest -q
git diff --check
```

Run focused tests during implementation and the proportionate full suite before
review. Documentation build commands are defined in the documentation setup.
