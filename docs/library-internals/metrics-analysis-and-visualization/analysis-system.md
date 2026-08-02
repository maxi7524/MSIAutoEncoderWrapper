# Analysis system

The current autoencoder analysis package is a provisional implementation and is
scheduled for structural revision.

## General abstraction

### Intended analysis role

Analysis answers a defined analytical question by selecting prepared data,
invoking metrics, aggregating results, and requesting reusable visualizations.
It should not own duplicate metric formulas or generic plot implementations.

### Prepared state

Preparation materializes only requested retained arrays and estimates memory
before allocation. Single-model and multi-model analyses share base context and
result records.

## Detailed implementation

### Current modules

[`analysis/autoencoder`](../../../src/msi_autoencoder_wrapper/analysis/autoencoder/__init__.py)
contains preparation/results, reconstruction, latent, heads, and binning
subdomains. `AutoencoderAnalysis` exposes convenience delegation; subdomain
objects implement individual analytical operations.

### Rewrite status

Some numerical and visualization functions are duplicated between analysis
subpackages and global `metrics`/`visualization`. The rewrite must move reusable
calculation and rendering outward, leaving analysis with selection, aggregation,
comparison, and result composition.

No new stable internal contract should depend on the current analysis object
layout until this TODO is completed.
