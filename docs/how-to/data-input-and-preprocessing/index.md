# Data input and preprocessing

These guides describe how spectra and their metadata enter the processing
pipeline before model execution.

## Contents

- [Configure MSI readers](readers.md) — select PyImzML or m2aia reading and inspect reader capabilities.
- [Configure spectral binning](binning.md) — map irregular spectra to a shared axis and select inverse-binning behavior.
- [Configure normalization](normalization.md) — define ordered scalar normalization and reconstruction output space.
- [Attach molecular annotations](annotations.md) — select annotations from the workspace SQLite catalog or paired local METASPACE CSV exports.

```{toctree}
:hidden:

readers
binning
normalization
annotations
```
