# Analysis infrastructure verification

Date: 2026-10-13. Branch: `tests_ae_and_new_binning`. Baseline HEAD: `5e3bc468a0e377badf88d52ee8482d3f7cf1e5a3`; working tree contains the scoped implementation. Generated provenance records source hashes and the tracked diff. No model training was performed. The original infrastructure verification did not run encoder inference; the subsequent CUDA spectral extension evaluates six existing checkpoints.

## Earlier infrastructure checks (before the CUDA-only correction)

| Check | Actual result |
| --- | --- |
| New focused tests, including actual CUDA | 32 passed |
| Earlier focused integration set including existing spatial/head/reconstruction tests | 66 passed, 1 CUDA skip inside device-restricted sandbox |
| Full analysis directory | 453 passed, 7 skipped, 2 failed, 33 setup errors |
| Independent baseline reproduction of the two affected existing test modules | Same 2 failures and 33 setup errors; 19 passed, 2 skipped |
| Numerical report on GTX 1060 6GB | All five checks passed |
| Real source decoding | Poseidon, Cardinal and BlockSAM decoded/cached successfully |
| Notebook execution | Both notebooks executed successfully |
| Python compilation | Passed |
| `git diff --check` | Passed |
| `uv lock --check --offline --cache-dir /tmp/msi-uv-cache` | Passed |

The broad-suite failures are reproducible on an isolated extraction of the unchanged baseline HEAD. `contractive_precompute` tests place settings outside the repository despite the implementation's repository-local requirement; `predictive_precompute` references a missing `14_09_26_predictive_final/vpu_precision_sweep.yaml`. The current task does not repair those earlier analysis contracts. The suite is not reported as passing.

No formatter, linter or static type checker is configured in pyproject.toml. None was introduced. A full repository suite, Sphinx build and package build were not run; focused functional/numerical tests, the analysis suite, notebook execution, compilation and lock verification cover the changed analysis surface.

## Numerical evidence

Environment: PyTorch 2.7.1+cu118, CUDA 11.8, NVIDIA GeForce GTX 1060 6GB. Device access requires execution outside the restricted sandbox; no silent CPU fallback was used for CUDA verification.

- Masserstein against independent SciPy float64 Wasserstein: maximum absolute error CPU `1.819594e-6`, CUDA `9.996427e-7`.
- InfoNCE CPU/CUDA gradient maximum absolute difference: `7.450581e-8`.
- Frozen float32 tolerance: `atol=1e-5`, `rtol=1e-4`.
- Float64 Masserstein gradcheck: passed with `eps=1e-6`, `atol=1e-5`, `rtol=1e-3`, away from zero cumulative imbalance.
- Gradient snapshots distinguish missing, zero, nonfinite and nonzero gradients; scaled norm computation is checked at magnitude 1e200.
- Latent rank: exact constant/rank-one controls, rotation invariance and scale invariance up to 1e100.
- Segmentation: exhaustive small matching oracle, analytical boundaries, row/label permutation invariants, unknown holes and explicit noise penalties.
- Neighbours: brute-force physical-distance oracle, radius equality, excluded-population and compact merged-map integration checks. No cross-source edges are possible between separate indices.
- Reconstruction: perfect/zero spectra, missing rare peak, worst-error identity, transport scale invariance and rejection of unrepresentable source intensities.

These checks validate the admissible fixtures and implemented contracts, not every possible floating-point input. Mutation testing, formal proof and performance-optimality proof were not executed.

## Data and mask evidence

Selected canonical images: Poseidon 4123×54; Cardinal MH0204_33 1560×10200, 811 diagnostic labels; BlockSAM NL_0705A190c 716×613, 716 normal labels. Arrays preserve numeric precision and original source identity. Persisted cache hashes were checked when reloaded. BlockSAM warnings about unsupported class constructors are retained; the actual slot fields are decoded and validated.

Poseidon manual trace: 29 polygons, six disjoint labels, 853886 available H&E pixels out of 2649728 image pixels after conflict/uncertainty exclusion. Saved source indices are row-major H&E image indices. The JPEG was visually inspected. This verifies the artifact construction and visible alignment only; contour accuracy has not been independently adjudicated, and MSI registration remains pending. No quantitative anatomical claim is made about a real model.

## Task conformance

| Criterion | Verdict | Evidence |
| --- | --- | --- |
| AC1 descriptions | PASS | Versioned dataset_cards/ with identities, units, labels and input limitations |
| AC2 source adapters | PASS | evaluation_data.py, three actual decoded caches and dataset_inventory.csv |
| AC3 indexed masks | PASS | reference.py, mask round-trip/hash/schema/order tests |
| AC4 reviewable manual trace | PASS | poseidon_regions.json, lossless H&E mask and comparison JPEG; MSI status explicitly pending |
| AC5 anatomical metrics | PASS | reference_metrics.py and independent matching/boundary controls |
| AC6 diagnostics | PASS | Existing metrics reused; rare tails, chemical profiles, latent spectrum and gradient checks |
| AC7 script reports | PASS | reference_validation_precompute CLI and reports/plots/numerical_values artifacts |
| AC8 thin notebooks | PASS | Two executed part_0 notebooks reading saved artifacts only |
| AC9 CPU/CUDA verification | PASS | 32 focused tests and five numerical report checks |
| AC10 preservation | PASS | No model/split/seed rewrite, no agent branch; user's existing notebook left untouched |

Overall task conformance: PASS for the scoped analysis infrastructure. Algorithm verification: PASS on the documented executable contracts and fixtures; real histology registration and pretrained input compatibility are NOT VERIFIED.

## Complexity and limits

- Mask identity join: O(N log N) sorting and O(K log N) lookup; O(N+K) auxiliary storage.
- Polygon rasterization: O(H*W*V) worst-case, bounded by polygon boxes; O(H*W) auxiliary storage.
- Reference contingency/matching: O(N log N + R*C + min(R,C)^2*max(R,C)); O(N+R*C) storage.
- Raster boundary distances: O(H*W) time/storage.
- Source-neighbour index: O(N log N) construction, output-sensitive queries with O(N) worst case; O(N) index storage.
- Reconstruction diagnostics: O(N*M) residual computation plus top-error selection, O(N*M) current auxiliary storage. Source matrices and existing metrics are retained; block processing would reduce peak memory for larger images.
- Latent SVD: O(N*D*min(N,D)); O(N*D) storage. A streamed covariance path could reduce memory when N is much larger than D.
- Source R decoding materializes the serialized release once; canonical NPY arrays are memory-mapped for reuse.

Optimality is unknown for the complete workflow. The index and dense per-feature diagnostics are plausible near-optimal for the required output; no optimality proof is claimed. HDBSCAN/GMM/K-means scaling depends on their configured estimator implementations. The demonstrated source samples and synthetic inputs are small; no large-dataset GPU memory claim is made.

## Reproduction commands

```bash
MPLCONFIGDIR=/tmp/msi-mpl-cache python -m pytest tests/analysis/test_pixel_reference.py tests/analysis/test_reference_metrics.py tests/analysis/test_evaluation_data.py tests/analysis/test_reference_validation_workflow.py -q
MPLCONFIGDIR=/tmp/msi-mpl-cache python -m pytest tests/analysis -q
python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_validation_precompute --settings assets/experiments/autoencoder_architecture/notebooks/agent_analysis_validation/analysis_settings.yaml --analysis datasets
python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_validation_precompute --settings assets/experiments/autoencoder_architecture/notebooks/agent_analysis_validation/analysis_settings.yaml --analysis demo
uv lock --check --offline --cache-dir /tmp/msi-uv-cache
```

Use the cu118/evaluation environment documented in README.md. Reports and masks inherit external_test_only; the APIs prohibit training/fit/selection purposes. A global interception of arbitrary legacy training loaders has not been implemented. Thresholds/cluster parameters used here are verifier fixtures; freeze production parameters separately after repairing and selecting pretraining.

## CUDA spectral extension

The user explicitly required CUDA-only model checks and clustering. Earlier CPU clustering/float64 inference artifacts were moved to `part_0_analysis_validation_results/archive/spectral_cpu_float64`; they are not the current comparison. No further CPU model reference checks were run after the correction.

- `.venv/bin/python -m pytest tests/analysis/test_gpu_partition_methods.py -q`: **18 passed**, actual CUDA access, 1.59 s.
- Tests cover exact analytical class recovery by all three methods, seeded repeatability, constant inputs, density components and noise, nonfinite inputs/settings, rank-one/constant/large-scale latents, TIC, evaluation flag/weight preservation, and float32 distance-overflow rejection.
- CUDA methods: K-means++ (five initializations), diagonal Gaussian EM, DBSCAN. The latter replaces CPU HDBSCAN and is a distinct analysis pipeline. Radius rule: 0.95 quantile of 12-neighbour distances, including self, fixed before scoring.
- Original float32 tolerances remain `atol=1e-5`, `rtol=1e-4`. A preliminary bit-exact GPU repeat check failed at maximum absolute difference `2.7939677e-9`; its log is preserved. Bit-exact equality is not claimed for real checkpoints.
- Spectral GPU comparison: **378 method/representation evaluations**, all `torch_cuda`, float32; all three methods report convergence. Eighteen fixtures span three layouts, two noise levels and three input seeds; each is evaluated directly and through six checkpoints.
- Actual checkpoint checks: **18 passed** (12 repeated-output checks plus six state/checkpoint preservation checks). Maximum GPU repeat difference: `3.725290298461914e-9`.
- Both updated notebooks executed successfully, using saved artifacts only: four code cells/four embedded images in part_0_1; eight code cells/32 embedded images in part_0_2.
- All 55 figure links in the spectral Markdown report and all three dataset preview links exist. Notebook schemas validate, all code cells have execution counts, and no error outputs are present.
- Compilation and `git diff --check` pass. No full CPU-based suite was rerun after the CUDA-only instruction; earlier broad-suite failures remain documented above.
- The user-owned rare-classes notebook SHA256 remains `a1cbec1b801774d1b5114f0820a1f6619878161d40fd99cf295d6d10a7607f36`.

### Extension conformance

| Criterion | Verdict | Evidence |
| --- | --- | --- |
| EX1 same-input CUDA methods | PASS | K-means, diagonal GMM, DBSCAN; explicit backend/dtype in every score row |
| EX2 spatial geometries | PASS | Two stripes, five stripes, five concentric regions; saved inputs/indices |
| EX3 real trained representations | PASS | Three existing checkpoints per family, strict loading, axis/TIC checks and preserved hashes |
| EX4 inspectable reports | PASS | 18 selected comparison/error/profile sets, CSV/NPZ, Markdown |
| EX5 external masks/previews | PASS | Three dataset JPEGs and separate provisional H&E annotation; missing MSI anatomy remains explicit |
| EX6 thin executed notebooks | PASS | Saved-figure/table readers, successful execution and artifact/schema checks |

Current source/clustering results establish behavior on the documented synthetic fixtures only. The fixed DBSCAN radius rule merges high-noise regions in some fixtures; it must not be represented as successful biological separation. The current generic legacy bundle/demo helpers still contain archived CPU estimator paths; only `spectral_partitions` is the CUDA-only executed workflow documented here.
