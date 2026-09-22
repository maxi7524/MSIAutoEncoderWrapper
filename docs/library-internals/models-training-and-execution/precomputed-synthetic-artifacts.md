# Precomputed synthetic artifacts

This document describes construction, persistence, and batch-time rendering
for artifact-backed synthetic pretraining selected by
`pretraining.kind: precomputed_synthetic`.

The ordinary generated-spectrum path is described in
[Synthetic pretraining internals](synthetic-pretraining.md). Campaign-specific
population choices remain next to their experiment configuration.

## Runtime boundary

`build_precomputed_synthetic_partitions()` is the boundary between a real
dataset and the trainer's synthetic phase. It parses `PrecomputedSyntheticConfig`,
computes a request fingerprint, obtains an immutable artifact from
`SyntheticArtifactStore`, and returns `PrecomputedSyntheticDataset` instances
for train and validation. The returned datasets preserve the existing
`SpectrumBatch` interface.

The real dataset supplies the active binner, molecular target schema, mapped
annotation index, and real training partition. The precompute path never uses
validation or test annotations to select synthetic molecular components. The
train partition is used while constructing the artifact; batch rendering then
reads the artifact alone.

## Artifact construction

### Source summary

For `peak_source: annotation`, the builder reads sparse entries from the mapped
annotation index for source IDs in the real training partition. It constructs
unique eligible bins for every molecular target, an occurrence count for every
target, and target IDs that co-occur with another target in a source-spectrum
bin. Entries outside the active axis and target vocabulary are discarded.

The summary stores unique target/bin geometry rather than one Python object per
source annotation. `peak_source: candidate_catalog` instead derives the source
from the loaded candidate catalog and records its path, size, and modification
time in the cache identity.

### Theoretical basis

`SyntheticPrecomputeBuilder` converts source geometry into immutable labelled
prototypes. Each prototype has a molecular target ID and an anchor bin. With
`isospec_envelope`, one theoretical profile is rendered per source class and
its anchor is associated with every bin available to that class. Other
renderers produce a profile per class/bin coordinate.

Every profile is rendered once, checked for finite non-negative values and a
positive total intensity, TIC-normalized, and stored as COO triplets:
`basis_rows`, `basis_columns`, and `basis_values`. The artifact stores only
non-zero profile entries, not a dense population-by-axis matrix.

### Population manifest

Each named population is a `SyntheticManifest`. Its slot-aligned arrays have
shape `(N, K)`, where `N` is the number of synthetic rows and `K` is the
maximum component count after right-padding.

| Array | Meaning |
| --- | --- |
| `component_ids` | Labelled sparse-basis prototype ID; `-1` otherwise. |
| `blank_centers` | Generic blank-peak anchor; `-1` otherwise. |
| `requested_target_indices` | Molecular class requested by a labelled token. |
| `component_kinds` | Single, base-class quota, blank quota, overlap bonus, rare bonus, or padding provenance. |
| `anchor_bins` | Stable representative bin per row. |
| `annotated_concentrations`, `blank_concentrations` | Per-row Dirichlet concentrations. |

Validation rejects a slot that is both labelled and blank, a row without a
component, a target request on a non-labelled slot, and non-positive or
non-finite concentrations. Schema version `3` is required because earlier
artifacts cannot represent multiple blank components in one row.

## Population compilers

### Axis coverage

`axis_coverage` emits `repetitions_per_bin` rows for every active-axis bin. An
annotated anchor includes all prototypes mapped to that anchor. A blank anchor
contains one generic blank component. This population provides exact bin
coverage and full molecular negatives for blank rows.

### Uniform mixture

`uniform_mixture` remains for compatibility. It selects non-empty anchor bins
until each has been selected `repetitions_per_bin` times, partitions selections
between `min_fragments` and `max_fragments`, and can drop components using
`detection_probability`. It contains labelled components only; blank-bin quota
sampling belongs to `class_quota_mixture`.

### Class-quota mixture

`class_quota_mixture` creates a token bag before creating spectra. The base bag
receives `class_quota` tokens for every eligible target and
`blank_bin_quota` tokens for every blank axis bin. Targets in the measured
multi-label overlap set receive `overlap_bonus_per_class` additional tokens.
The least frequent `ceil(rare_fraction * eligible_target_count)` targets
receive `rare_bonus_per_class` additional tokens. Bonuses are additive.

The bag is seeded, shuffled, and partitioned into rows whose inclusive sizes
are between `min_fragments` and `max_fragments`. A requested class selects a
prototype from that class's sparse basis bank. The compiler balances exact
token counts across the complete manifest; it does not impose a uniqueness
constraint on prototype IDs inside an individual row. Molecular target
construction later treats repeated class IDs as one positive class.

For each quota population, the annotated Dirichlet concentration is

```
max(minimum_annotated_concentration,
    blank_concentration * blank_token_count / annotated_token_count)
```

when both token categories occur. This calibrates aggregate expected annotated
and blank mass against the compiled population while retaining the configured
annotated lower bound.

### Combined population

`combined` takes two or more existing manifests, pads their slot arrays to a
shared width, concatenates their rows and concentration vectors, then applies
a deterministic population-specific permutation. It does not reconstruct
profiles or alter token quotas. A cycle in named combined populations is
rejected during artifact construction.

## Persistence and cache identity

The store directory contains one directory per safe artifact key and
fingerprint prefix. A complete artifact consists of `artifact.npz` and
`metadata.json`. The metadata records schema version, artifact key, full
fingerprint, normalization, blank profile radius, and manifest names.

`SyntheticArtifactStore.load_or_build()` first attempts a matching load. On a
miss, it acquires an exclusive lock for that key/fingerprint pair, checks again
after acquiring the lock, builds in a temporary directory, writes all arrays,
and atomically publishes the directory with `os.replace`. A missing array,
partial directory, schema mismatch, or fingerprint mismatch is a cache miss.

The fingerprint includes schema version, artifact configuration, renderer and
renderer parameters, normalization, blank radius, seed, target names,
population declarations, active axis, and source identity. Annotation sources
also contribute sparse annotation-index arrays and selected train source IDs.

## Batch-time rendering

`PrecomputedSyntheticDataset.__getitem__()` returns only an integer manifest
row ID. `collate_fn()` performs numerical work for the requested batch. For a
batch size `B`, maximum manifest width `K`, prototype count `P`, and axis width
`M`, the data flow is:

```text
row IDs
  -> component and blank slots                         (B, K)
  -> row-stable Dirichlet weights                      (B, K)
  -> sparse prototype composition                      (B, P)
  -> sparse basis multiplication                       (B, M)
  -> generic blank profiles and normalization          (B, M)
  -> SpectrumBatch values, targets, masks, metadata
```

The dense tensor exists only for the current `(B, M)` batch. COO basis values,
the composition matrix, intermediate spectra, and targets are rendered in
`float32`; the `SpectrumSpace` axis is explicitly converted to the dataset
dtype. This prevents an inherited `float64` axis from widening the batch
pipeline.

Blank slots add normalized triangular profiles at declared anchors. The
renderer supports `tic`, `max`, `l2`, and `none` normalization. It clamps the
denominator to the smallest normal value for the working dtype and rejects
non-finite or negative output.

## Targets, randomness, and validation rows

The molecular value matrix has shape `(B, C)` and starts at zero. Every
labelled prototype contributes its target class through `scatter_add`; values
are clamped to binary presence. The molecular target mask is fully true:
classes absent from the synthetic composition are known negatives. The
simulated-negative mask is true exactly for those absent classes. Other target
schemas remain masked.

Dirichlet weights use an independent NumPy generator seeded by
`(seed, epoch, manifest_row_id)`. Changing DataLoader batch composition or
order therefore does not change a row's spectrum for a fixed epoch.
`set_epoch()` changes training weights without changing the static manifest.
The validation dataset selects evenly spaced manifest rows, sets `fixed_epoch`,
and has invariant synthetic intensities across training epochs.

## Operational limits

The artifact eliminates repeated isotope-envelope rendering and avoids a dense
`N × M` precomputed corpus. It does not eliminate the dense `(B, M)` tensor
needed by the model. Cache construction still requires the source dataset,
binner, target schemas, annotation index, and real train partition. A cache hit
avoids rebuilding the annotation summary and basis but the caller still
constructs the dataset context.

Focused behavioral contracts are in
[`tests/data/test_precomputed_synthetic.py`](../../../tests/data/test_precomputed_synthetic.py)
and campaign expansion contracts are in
[`tests/runtime/test_precomputed_campaign_configuration.py`](../../../tests/runtime/test_precomputed_campaign_configuration.py).
