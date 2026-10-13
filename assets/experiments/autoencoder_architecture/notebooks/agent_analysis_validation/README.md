# Agent analysis validation

This folder demonstrates the script-produced analysis contract before implementing model ablations. The maintained implementation lives in `analysis/autoencoder/{spatial,reconstruction,latent,experiments}`. Notebooks only read canonical tables and figures. No model training occurs. The spectral comparison evaluates six existing trained checkpoints on synthetic spectral images; the notebooks only display saved results.

## Entry points

- [Dataset cards](dataset_cards/poseidon_ccrcc.md): [Cardinal](dataset_cards/cardinal_rcc.md), [BlockSAM](dataset_cards/blocksam_kidney.md), [Poseidon](dataset_cards/poseidon_ccrcc.md).
- [Settings](analysis_settings.yaml): shared paths, device, seeds and fixed verification parameters.
- [Task specification](TASK_SPEC.md): acceptance criteria and scope.
- `part_0_1_reference_datasets.ipynb`: source indices and provisional H&E annotation.
- `part_0_2_model_analysis_contract.ipynb`: numerical verification and the future Markdown report format.
- [Verification report](VERIFICATION.md): executed checks and remaining limitations.

## Run

Raw source adapters require Python >=3.11 and the optional `evaluation` extra. Cached numeric readers do not require R or rdata. Use the cu118 build for the local GTX 1060.

```bash
uv sync --extra cu118 --extra evaluation
python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_validation_precompute --settings assets/experiments/autoencoder_architecture/notebooks/agent_analysis_validation/analysis_settings.yaml --analysis datasets
python -m msi_autoencoder_wrapper.analysis.autoencoder.experiments.reference_validation_precompute --settings assets/experiments/autoencoder_architecture/notebooks/agent_analysis_validation/analysis_settings.yaml --analysis spectral_partitions
```

`spectral_partitions` requires CUDA and refuses CPU fallback. It performs inference, mixture generation, clustering and small latent-rank calculations in float32 on the GPU. File I/O, scoring exported integer labels and plotting use host utilities. The older `demo` command includes CPU reference calculations and is retained only for historical verifier controls; do not use it for this GPU protocol. `masks` rebuilds the manual trace without CUDA. Dataset decoding is cached under ignored `data/`; synthetic examples and reports are under ignored `part_0_analysis_validation_results/`:

```text
part_0_analysis_validation_results/
├── reports/                 # Markdown with relative figure links
├── plots/                   # Reference preview and metric distributions
├── numerical_values/        # CSV, strict JSON, recorded checks and controls
├── cases/<fixture>_seed_<s>/ # Complete model-report examples
└── provenance_<analysis>.json
```

Both notebooks share the same canonical report outputs; their corresponding results directories link to those outputs. Re-executing a notebook does not decode R files, load models or launch GPU computations.

## Interpretation

Metric priority: segmentation → molecule/class precision → reconstruction preservation → latent/gradient stability. Scoring includes reference coverage, noise, region recall/precision, Dice/IoU, ARI/AMI, confident interiors and registered-raster boundaries. Matching is rectangular and score-only. Current spectral comparisons use CUDA K-means, diagonal-covariance GMM and DBSCAN. DBSCAN uses a frozen neighbourhood-distance quantile for its radius, independently of reference labels. Archived CPU controls use HDBSCAN and full-covariance GMM; Leiden and UMAP are deferred until the clustering protocol is agreed. The synthetic K=2 is not a recommendation for real tissue.

Rare-feature membership uses prevalence frozen from ordinary training/validation; it must not be redefined on external test. Reconstruction includes distributions, q95/q99/max, positive-only relative errors, missed rare features and individual worst failures. Zero-target relative error is unavailable; absolute errors still record hallucinations. The existing spectral-angle metric assigns pi/2 to a zero-norm pair; this inherited convention is documented rather than silently changed.

The H&E mask is provisional and has six disjoint labels plus unknown. Review `plots/manual_annotation_comparison.jpeg`. Its row-major image indices cannot be joined to MSI row indices until independently checked registration is available. No classifier/SVM is fitted on external anatomical labels.

## Retained model output contract

`--analysis bundle --image-cache <canonical-cache> --prediction-bundle <outputs.npz>` accepts arrays with `dataset_id`, `image_id`, `pixel_ids`, `latents`, `reconstruction`, optional `molecule_logits`, `molecule_targets`, `molecule_availability` and `gradient_<component>` snapshots. The dataset and image identities and exact pixel order are verified. Production settings must supply `model_context`, `model_input_representation`, frozen `reference_prevalence`, thresholds and prespecified clustering parameters. The example settings intentionally do not invent these model-dependent quantities.

All arrays must describe the same frozen evaluation task. Source representation compatibility is checked before the CLI evaluates a bundle. Model-specific binning/normalization, polarity, vocabulary and checkpoint validation remain the responsibility of the inference adapter; this demonstration does not assert that a raw source release is inference-ready. External-only restrictions are enforced by these evaluation APIs; no global training-loader interception has been added.

## Next model stages

First repair/re-run the meaningful pretraining comparisons and choose sampling quantities/combinations, including spectrum permutations. Freeze the selected pretraining settings. Preserve and continue training the pretrained head; do not reset or freeze its weights. The user reports that head freezing destroys prediction quality and that the earlier analysis was broken; historical rankings are not accepted evidence.

Then implement the selected chemical graph and three VPU constructions (direct classifier, residual MLP over molecule/class logits, richer latent/context residual). Confirm the precise third construction against the transcripts before implementation. Next spatial neighbourhood, then variance/covariance redundancy. Every evaluated model condition requires at least three independent training repetitions. Existing population/split/seed machinery is reused. This task adds evaluation infrastructure and numerical verification, not a new training campaign or agent branch.

## Spectral comparison artifacts

`reports/spectral_partition_comparison.md` contains side-by-side method panels, error maps and observed/decoded spectral profiles. `numerical_values/spectral_partition_{scores,regions,summary}.csv` retain every seed/checkpoint/method. Exact input observations, source pixel IDs and latent outputs are retained in NPZ files. The manifest records checkpoint/config hashes, axis, dtype and backend. Three existing checkpoints per model family are evaluated; no new training repetitions are launched.

Every selected external dataset has a `reference/source_signal_and_mask.jpeg`, also linked from `part_0_1_reference_datasets.ipynb`. Cardinal labels are diagnostic; the selected BlockSAM acquisition contains only normal tissue. Poseidon has no accepted numeric MSI anatomy mask; its provisional H&E trace is shown separately.

Historical CPU clustering/float64 inference artifacts are preserved under `archive/spectral_cpu_float64/`. They are excluded from the current GPU result tables.
