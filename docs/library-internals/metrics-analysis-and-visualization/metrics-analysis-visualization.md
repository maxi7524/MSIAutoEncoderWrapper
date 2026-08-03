# Metrics, analysis, and visualization

Numerical computation, analytical orchestration, and rendering are separate
layers even when one user operation invokes all three.

## General abstraction

### Metrics

Metrics accept numerical representations and return numerical results. They own
domain requirements such as non-negative inputs, linear intensity semantics,
mass-axis compatibility, and reduction behavior.

### Analysis and visualization

Analysis selects data and combines metrics for a question. Visualization accepts
already prepared values and theme configuration and returns figures or views.

## Detailed implementation

### Numerical layer

[`metrics`](../../../src/msi_autoencoder_wrapper/metrics/__init__.py) registers spectral,
classification, class, embedding, and Masserstein strategies. Compatibility
checks compare metric requirements with normalization capabilities before
execution.

### Rendering layer

[`visualization`](../../../src/msi_autoencoder_wrapper/visualization/__init__.py) owns theme,
spectrum comparison, spatial views, metric plots, and interactive ion images.
Themes map model and class identities to stable colors.

### Analytical composition

Reconstruction analysis maps per-spectrum errors to spatial coordinates, ranks
spectra/features, and combines raw and reconstructed ion images. Latent analysis
combines projections with target labels. Head analysis combines probabilities,
masks, per-class measures, and spatial maps.

Reusable formulas and plots discovered during the analysis rewrite must move to
their global layers rather than remain hidden in analysis modules.
