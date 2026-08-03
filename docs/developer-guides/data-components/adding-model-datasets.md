# Add a model dataset

A model dataset defines sample identity, input representation, targets, and
partition hooks for one model workflow.

## Dataset scope and contracts

### Base contract

Inherit `MSIBaseDataset` and implement length, item access, stable sample IDs,
and any split target, mask, or group hooks used by supported strategies.

### Context ownership

Single-image datasets receive `active_context`; cohort datasets receive
`cohort_context`. Do not read workspace global state inside sample access.

## Implementation instructions

### Register and configure

Decorate with `DatasetManager.register_dataset(name)`. Store source, target,
normalization, and split state through `ConfigurableComponent`.

### Support efficient batching

Implement raw sample/batch access when the shared `BatchPreprocessor` should bin
and normalize on a selected device. Preserve target schemas and masks in both
single and native-batch paths.

### Add split support and tests

Return stable group and target values. Test indexing, negative indices, sample
identity, target mappings, missing annotations, split reproducibility,
configuration restoration, and cohort schema consistency.
