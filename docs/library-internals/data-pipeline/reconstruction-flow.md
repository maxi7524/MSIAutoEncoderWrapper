# Reconstruction flow

Reconstruction moves model decoder output through output-domain validation,
optional denormalization, and optional inverse-bin selection.

## General abstraction

### Dense decoder output

The decoder reconstructs the binner feature axis. A configured output activation
must enforce the intensity domain required by losses and metrics.

### Source-space output

Source-scale reconstruction requires the normalization trace from the input
batch. Inverse normalization occurs at the configured stage.

## Detailed implementation

### Decode and activate

Autoencoder decoder implementations apply the activation built by
[`output_activation.py`](../../../src/msi_autoencoder_wrapper/models/architectures/types/autoencoders/decoders/output_activation.py).
Non-negative MSI reconstruction uses an activation such as softplus.

### Restore scale

[`AutoencoderContextInterface.decode()`](../../../src/msi_autoencoder_wrapper/core/mixins/active_context/autoencoder_context_manager.py)
resolves call-level or context reconstruction policy and applies inverse
normalization where allowed by the trace capabilities.

### Select mass positions

When inverse binning is requested, the configured strategy selects dense bins
and returns their binner-axis mass coordinates. It cannot recover raw peaks lost
during forward aggregation.
