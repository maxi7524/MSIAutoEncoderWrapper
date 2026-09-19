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

### Annotation population

`AnnotationPopulation` is separate from `IonCatalogue`. It retains complete
train-only records in the form `(source spectrum ID, binned coordinate,
molecular target columns)`. This preserves multiple annotations at one bin in
one pixel and distinguishes them from the same bin in another pixel. A missing
record for a selected train pixel/bin is an explicit complete molecular
negative for synthetic BCE.

### Sampling strategy

`SyntheticSamplingStrategy` converts a local NumPy generator and a
`SyntheticSamplingContext` into `SyntheticSampleDefinition`. The definition
contains peak centres, optional target-column identities, and the target-mask
policy. Concrete strategies are registered by `SyntheticSamplingManager` and
discovered from `data/pretraining/strategies`.

Pixel-aware annotation strategies are organized under
`data/pretraining/strategies/annotations`, with one module per strategy and
shared selection helpers in `annotations/common.py`. The axis-coverage strategy
uses its plan position to visit every active bin equally often. Mixture
strategies select complete source-local records, so their target is the union
of all component labels. Before each epoch, mixture strategies allocate their
requested primary classes through a deterministic quota schedule; requested
class counts differ by at most one. Genuine multi-label overlaps can still add
secondary positives, which are retained in per-sample generation metadata.

### Spectrum representation

`SyntheticRepresentationStrategy` converts a declared component composition
into a dense binned spectrum. Renderers are independently registered and may
be selected globally for a phase or overridden by one sampling-plan entry. The
initial `triangular_peak` implementation preserves the historic synthetic peak
profile. `isospec_envelope` is the molecular renderer: it resolves the final
formula/adduct composition, enumerates fine isotope lines with IsoSpec, applies
the signed electron-mass correction, maps every line using the active binner,
and accumulates collision bins. It adds independently drawn component
occupancy and response weights before the dataset-level normalisation. The
renderer requires a declared molecule, so plans must override unlabelled axis
coverage or background entries to a non-molecular renderer. Rendering does not
alter annotation sampling or BCE target construction.

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

Phase snapshots make a synthetic pretrain reusable inside one trainer run.
`save_model_state_as` captures the post-phase state after optional best-checkpoint
restoration. `restore_model_state_from` loads that immutable state before freeze
configuration and optimizer construction for a later phase. This supports paired
frozen- and unfrozen-head branches without repeating pretraining. Runtime
continuation checkpoints persist these snapshots together with model and
optimizer state.

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
