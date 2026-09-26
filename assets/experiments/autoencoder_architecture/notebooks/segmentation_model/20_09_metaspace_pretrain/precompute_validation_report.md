# Kidney synthetic-precompute validation

Date: 2026-09-20

## Scope

This is a historical bounded real-data probe of an earlier revision of the
artifact-backed `axis` phase. It is not a synthetic fixture and it does not
validate or run the current 310-task campaign.

Input data:

- image: `data/kidney_workspace/datasets/kidney/kidney.imzML` and `.ibd`;
- spatial population: the real coordinates `x=1..10`, `y=1..10`, `z=1`
  (100 source spectra);
- annotation policy retained 37 annotated spectra;
- m/z range: 150.0--200.0 with a 0.55 step (91 bins);
- selected population: `axis`, five repetitions per bin;
- rendered population: 455 samples.

The selected spatial rows are passed to the normal dataset/annotation path as
source-ID ranges. No CSV, temporary synthetic data, or dense `N x M` corpus is
used.

## Numerical result

Historical command (the phase is named `synthetic_single` in the current YAML;
this exact invocation applies to the earlier revision only):

```bash
UV_CACHE_DIR=/tmp/pep-compass-uv-cache uv run --extra cu118 python \
  assets/scripts/benchmarks/benchmark_precomputed_synthetic.py \
  --config assets/experiments/autoencoder_architecture/experiment_runs_configs/segmentation_model/20_09_26_metaspace_base_pretrain/pretraining_experiment.yaml \
  --phase-name synthetic_axis_pretraining \
  --spatial-side 10 --mass-min 150 --mass-max 200 --batches 16
```

| Quantity | Observed value |
| --- | ---: |
| First preparation, persisted artifact already present | 8.097 s |
| Matching in-process cache load | 3.249 ms |
| Rendered samples | 455 |
| Batch rendering wall time | 9.295 ms |
| Batch rendering throughput | 48,949 samples/s |
| Incremental process peak RSS | 304,033,792 B (290 MiB) |
| Persistent artifact size | 2,830 B |
| Sparse basis prototypes | 1 |
| Non-finite spectra | 0 |
| Negative intensities | 0 |
| Maximum TIC error | 0.0 |
| Unknown molecular targets | 0 |

The first cold build before persistence took 8.359 s; after the artifact was
written, the same invocation loaded it in 8.097 s total, dominated by opening
the imzML and constructing the real annotation view. The artifact itself loads
in 3.249 ms within that prepared dataset.

## GPU check

The local non-sandboxed command using `uv run --extra cu118` found and executed
on `NVIDIA GeForce GTX 1060 6GB` (`cuda:0`). The sparse synthetic dataset is
deliberately rendered in the CPU DataLoader; training subsequently transfers
the completed `SpectrumBatch` to the model device. This probe validates the
precompute/data boundary, not a model optimisation step.

## Permutation probe

The 10x10 crop on 150.0--200.0 had fewer than 15 distinct nonempty annotated
m/z bins. The configured `uniform_mixture` (`min_fragments: 15`) correctly
rejected it; this is a configuration guard, not a memory or numerical failure.

A second real-data probe used the same source-ID method, a 30x30 crop (900
source spectra, all 900 retained after annotation policy), and the production
200.0--900.0 axis (1,273 bins). It passed:

| Quantity | Observed value |
| --- | ---: |
| Sparse basis prototypes | 94 |
| Precomputed permutation samples | 124 |
| First preparation | 11.911 s |
| Matching cache load | 7.809 ms |
| Batch rendering throughput | 3,018 samples/s |
| Incremental process peak RSS | 304,250,880 B (290 MiB) |
| Maximum TIC error | 1.1920929e-7 |
| Non-finite or negative spectra | 0 |
| Unknown molecular targets | 0 |

## Historical numerical stress test

The decoder/objective path of the then-current configuration was executed in
`float32` on the local `NVIDIA GeForce GTX 1060 6GB` using `uv run --extra cu118`.
It used three stride-3 convolution stages, a `Softplus` decoder, TIC
normalization with `epsilon: 1e-12`, both production axes, the configured
MSE/Masserstein reconstruction paths, and 508 molecular BCE logits. This is
not a full numerical validation of the current resolved class mapping.

The test deliberately used a batch of 16 high-amplitude latent vectors
(`z ~ N(0, 12^2)`) and high-amplitude molecular logits (`N(0, 25^2)`) before a
full backward pass. This is a stronger overflow/underflow check than the
normal initialization regime; it is not a convergence claim.

| Axis bins | Decoder spatial lengths | TIC maximum error | Output finite | All gradients finite |
| ---: | --- | ---: | --- | --- |
| 1,273 (200--900) | 1,273, 423, 139, 46 | 1.1920929e-7 | yes | yes |
| 5,273 (100--3000) | 5,273, 1,757, 584, 194 | 1.1920929e-7 | yes | yes |

The numerical protections exercised by this historical probe are: nonnegative
`Softplus` decoder outputs; TIC normalization that returns zeros instead of
dividing by a total intensity at or below epsilon; stable
`BCEWithLogitsLoss`; class-balance denominators clamped to at least one;
finite-output/loss checks; and gradient clipping at norm 5.0. The
Masserstein implementation consumes TIC-normalized nonnegative spectra and
uses cumulative sums, without materializing a transport matrix.

One monitored edge case remains: if every decoder value underflows to zero,
TIC normalization safely returns a zero spectrum (no NaN/Inf), but that sample
has no reconstruction gradient through normalization. This was not observed
in the stress test; log the fraction of zero-TIC predictions during the first
production run.

## Conclusion

The bounded axis and permutation probes passed. The artifacts are persisted,
reload without rebuilding profiles, produce finite nonnegative TIC-normalized
spectra, and supply complete molecular BCE labels. A full-axis benchmark
remains the final scale check before approving the entire production campaign.
