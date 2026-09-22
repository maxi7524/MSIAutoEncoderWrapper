# Kidney synthetic-pretraining campaign methodology

This document defines the experiment implemented by
[`pretraining_experiment.yaml`](pretraining_experiment.yaml). It specifies the
scientific comparison and the configured sample populations. Runtime and
artifact interfaces are described in the [synthetic-pretraining internals](../../../../../../docs/library-internals/models-training-and-execution/synthetic-pretraining.md);
general phase configuration is described in the [synthetic-pretraining how-to](../../../../../../docs/how-to/models-and-training/synthetic-pretraining.md).

## Cohort and axes

The campaign uses a proportional multilabel subset with fraction `0.2`, then a
`90% / 5% / 5%` train/validation/test split. The split and DataLoader seeds are
fixed at `42` and `43`, respectively. The datasets
`2024-02-20_01h45m20s`, `2024-02-20_01h46m58s`, and
`2024-02-20_01h54m41s` are excluded before subsetting and splitting. They are
not evaluated by this campaign; they are reserved for a later held-out analysis.

Every schedule is repeated five times on both active axes. The reconstruction
input, axis coordinates, model parameters, and synthetic representation use
`float32`.

| Axis | m/z range | Components per permutation spectrum |
| --- | --- | ---: |
| `axis-200-900` | 200--900 | 3--61 |
| `axis-100-3000` | 100--3000 | 4--68 |

The component bounds are explicit configuration values. They are not inferred
from the current data split, so rerunning the experiment does not change an
ablation because of a recomputed IQR.

## Synthetic populations

All population construction uses only molecular annotations available in the
real training partition. Molecular isotope profiles are constructed with
`isospec_envelope`. Blank bins have no molecular formula and are represented as
generic peaks.

### Single-bin population

`axis` contains exactly ten generated samples for every bin on the active
axis. A selected annotated bin carries all labels mapped to that bin. A blank
bin is retained as a known negative for every molecular label. This population
tests whether isolated bin position and its complete local target set carry
useful information.

### Permutation population

Permutation populations first compile an exact component-token bag and then
randomly pack it into spectra within the axis-specific bounds in the table
above. Their relative intensities are drawn from a Dirichlet distribution
followed by the configured response model. The current compiler does not
deduplicate prototype IDs within an individual spectrum; repeated IDs remain
one positive molecular target.

The base bag has exactly 30 occurrences of each molecular class and exactly
30 occurrences of every blank bin. The overlap and rare mechanisms add tokens;
they do not replace or reweight the base bag.

| Population modifier | Additional token quota |
| --- | --- |
| Base | none |
| Overlap | 30 additional occurrences for every class occurring in a genuinely multi-label training bin |
| Rare | 30 additional occurrences for every class in the least frequent 25% of training classes |

Annotated and blank components are mixed in every permutation spectrum. The
Dirichlet annotated concentration is `max(2, n_blank / n_annotated)` for the
compiled bag and the blank concentration is `1`. This makes annotated peaks
more likely to receive a substantial share when a spectrum also contains many
blank components, without removing blank-bin coverage.

## Pretraining ablation

Each synthetic phase runs for 10 epochs with Masserstein reconstruction loss
(weight `1.0`) and multilabel BCE (weight `0.2`). BCE uses complete synthetic
targets: absent classes are known negatives. The classifier always has one
`molecule_vpu` logit head; only its loss interpretation changes in real-data
adaptation.

| ID | Population | Optimizer-phase ordering |
| --- | --- | --- |
| P0 | single bins | one single-bin phase |
| P1 | base permutations | one permutation phase |
| P2-joint | single bins + base permutations | one shuffled joint phase |
| P2-staged | single bins, base permutations | two sequential phases |
| P3-joint | single bins + base permutations + overlap quota | one shuffled joint phase |
| P3-staged | single bins, then base permutations + overlap quota | two sequential phases |
| P4-joint | single bins + base permutations + rare quota | one shuffled joint phase |
| P4-staged | single bins, then base permutations + rare quota | two sequential phases |
| P5-joint | single bins + base permutations + overlap + rare quotas | one shuffled joint phase |
| P5-staged | single bins, then base permutations + overlap + rare quotas | two sequential phases |

In a joint schedule, complete precomputed single-bin and permutation rows are
concatenated and deterministically shuffled into one population; the DataLoader
shuffles its rows again each epoch. In a staged schedule, single-bin rows are
trained for 10 epochs before permutation rows are trained for another 10 epochs.
The permutation component-token bag is generated once and packed once, not
resampled into new spectra every epoch. The P2--P5 comparisons therefore change
both ordering and optimizer-step budget (one versus two 10-epoch phases); they
are not a pure ordering-only ablation. P1 omits single-bin rows.

## Real-data branches

The campaign also trains a real-only baseline for 10 epochs. Every synthetic
schedule materializes its post-pretraining model as an independent task. Two
dependent adaptation tasks load that exact persisted artifact and save their own
model weights.

| Branch | Real-data epochs | Head state | Purpose |
| --- | ---: | --- | --- |
| Pretraining-only real test | 0 | persisted pretraining task | evaluate synthetic pretraining without real adaptation |
| Frozen-head adaptation | 10 | `molecule_vpu` frozen | adapt the representation while retaining synthetic logits |
| Unfrozen adaptation | 10 | trainable | adapt both representation and logits |

All real-data branches use Masserstein reconstruction (`1.0`), VPU (`0.2`),
contractive Fisher--Rao spectral-hinge regularization (`2e-3`), and InfoNCE
(`1e-3`, temperature `0.07`). The contrastive negatives use multilabel Jaccard
weighting with overlapping-label negative weight `0.1`. Synthetic phases do
not use the contractive or contrastive terms.

## Campaign size and execution

There are 11 logical schedules per axis: the real-only baseline plus the 10
synthetic schedules in the ablation table. Every synthetic schedule expands to
three model tasks (`pretrained`, `frozen_head`, and `unfrozen_head`). With two
axes and five repetitions, this produces 310 planned model tasks: 10 real-only
models and 100 complete three-model pretraining groups. The YAML defaults to
`execution.backend: local`; the Entropy scripts materialize the same plan and
submit individual parent-ready tasks through Slurm.

The synthetic artifact is persisted under
`data/kidney_workspace/precompute/synthetic_pretraining`. Its fingerprint
includes the active axis, training-derived annotation population, population
configuration, representation configuration, and seed. A matching artifact is
reused; a changed input produces a distinct artifact rather than silently
reusing stale rows.

## Validation contract

Before a production run, validate the following invariants for each axis and
selected population:

1. Axis coverage has exactly `10 * number_of_bins` rows.
2. The base permutation bag contains exactly 30 tokens per class and per blank
   bin; overlap and rare quotas are additive.
3. Every permutation row has a component count within the configured manual
   bounds. Repeated prototype IDs are allowed; their molecular targets are
   combined into the row-level multilabel target.
4. Synthetic spectra, targets, and coordinates are finite `float32` tensors;
   spectral values are non-negative and TIC-normalized.
5. BCE targets include all labels at an annotated bin and known negatives for
   absent labels.
6. The joint population is shuffled, whereas staged phases preserve
   single-bin-before-permutation ordering.
7. A local CUDA forward/backward pass through the configured model and both
   objectives produces finite losses and gradients.

The focused artifact, planning, batch-pipeline, and phase tests exercise these
contracts. The separate `precompute_validation_report.md` in this directory is
a bounded empirical probe and is not the methodological definition of the
campaign.
