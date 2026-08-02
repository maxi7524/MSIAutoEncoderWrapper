# General model families

These instructions add a completely new architecture family, such as a
diffusion model, rather than another autoencoder implementation.

## Contents

- [Add a model family](adding-a-model-family.md) — register the master graph and family-owned component contracts.
- [Support single-image contexts](single-image-models.md) — connect the family to local datasets and runtime state.
- [Support cohort contexts](cohort-models.md) — retain multi-image identity and model policy.
- [Integrate runtime and persistence](runtime-and-persistence.md) — create functionality interfaces and artifact reconstruction.
- [Test a model family](testing-model-families.md) — verify contracts, assembly, execution, configuration, and loading.

```{toctree}
:hidden:

adding-a-model-family
single-image-models
cohort-models
runtime-and-persistence
testing-model-families
```
