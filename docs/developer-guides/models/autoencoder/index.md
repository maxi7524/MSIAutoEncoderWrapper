# Autoencoder extensions

These instructions modify the existing autoencoder family and its component
contracts.

## Contents

- [Add an autoencoder graph](adding-an-autoencoder.md) — define another autoencoder master implementation.
- [Add an encoder or decoder](encoders-and-decoders.md) — implement latent and reconstruction contracts.
- [Add projectors, heads, and presets](projectors-heads-and-presets.md) — extend auxiliary components and configuration macros.
- [Add synthetic sampling strategies](adding-synthetic-sampling-strategies.md) — register, configure, and test generated-spectrum strategies.
- [Test autoencoder components](testing-autoencoders.md) — verify shapes, non-negativity, targets, variational outputs, and round-trip.

```{toctree}
:hidden:

adding-an-autoencoder
encoders-and-decoders
projectors-heads-and-presets
adding-synthetic-sampling-strategies
testing-autoencoders
```
