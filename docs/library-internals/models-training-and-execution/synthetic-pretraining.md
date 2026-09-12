# Synthetic pretraining internals

Synthetic pretraining creates dense spectra on the active binned axis for one
training phase. It supplies the existing `SpectrumBatch` contract, so models,
criteria, DataLoaders, checkpointing, and phase transitions remain unchanged.

## Component boundaries

### Peak source

`SyntheticPeakSource` defines target names, binned feature count, and candidate
coordinates for every target column. `CataloguePeakSource` adapts the current
annotation-derived `IonCatalogue`. This boundary permits a future source based
on candidate molecules without coupling sampling strategies to an annotation
index.

### Sampling strategy

`SyntheticSamplingStrategy` converts a local NumPy generator and a
`SyntheticSamplingContext` into `SyntheticSampleDefinition`. The definition
contains peak centres, optional target-column identities, and the target-mask
policy. Concrete strategies are registered by `SyntheticSamplingManager` and
discovered from `data/pretraining/strategies`.

### Dataset and target construction

`SyntheticSpectrumDataset` renders triangular peak profiles, normalizes the
result, and creates values and masks for the configured target schemas. It owns
no production spectra. The dataset returns `SpectrumBatch` through its custom
collator.

## Phase execution

`MSIPyTorchTrainer` recognizes `phase.pretraining`, builds synthetic train and
validation partitions, and uses them only for that phase. The active real
dataset remains unchanged. Before a subsequent phase without `pretraining`, the
trainer restores the original real train, validation, and test partitions while
keeping the same model instance and its updated parameters.

For annotation-backed synthesis, `build_synthetic_partitions()` derives the
target vocabulary and binned coordinates from the active dataset. It selects
eligible molecular targets from positive, available labels in the real training
partition. This prevents validation and test labels from determining synthetic
positive-ion eligibility.

## Sampling and reproducibility

Each dataset item creates a local random stream from synthetic seed, epoch, and
item index. `set_epoch()` changes only that stream. The generator does not alter
global NumPy or Torch random state.

An explicit plan expands to exactly `samples` examples per epoch. The validation
plan is derived with largest-remainder proportional allocation and has exactly
`validation_samples` examples. Legacy `modes` remain a weighted random choice
when no explicit plan is supplied.

## Target semantics

Molecular targets are complete when `label_targets` is true: generated labelled
components are positive and absent target classes are negative. When it is
false, all target masks are false, so head criteria contribute zero while
reconstruction remains active. Chemical-class targets retain candidate-derived
uncertainty masks. Element counts are emitted only for a single labelled
component without background.

For configuration examples, see [Run synthetic spectral pretraining](../../how-to/models-and-training/synthetic-pretraining.md). For extension rules, see
[Add synthetic sampling strategies](../../developer-guides/models/autoencoder/adding-synthetic-sampling-strategies.md).
