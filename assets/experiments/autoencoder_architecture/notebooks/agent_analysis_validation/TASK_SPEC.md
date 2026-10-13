# Analysis validation task

## Scope and order

Implement external dataset descriptions/adapters, indexed anatomical references, segmentation-led analysis and a script-produced Markdown report before model ablations. Model work follows pretraining design/freeze → graph and VPU supervision → spatial objectives → redundancy. Preserve the current branch, existing split/seed machinery, graph analysis and user-owned notebooks. Do not train a model or implement graph architecture in this task.

## Acceptance criteria

- AC1: Document the three reference datasets, source identities, input compatibility, label provenance, coordinate conventions and unavailable information.
- AC2: Load their source objects and convert supported contents to an explicit canonical evaluation schema; reject unsupported structures rather than inventing spectra or coordinates.
- AC3: Persist/load pixel-indexed masks with vocabulary, identity, coordinate space, availability, hash and lineage; reject duplicate/missing/misaligned keys.
- AC4: Digitize visible Poseidon regions into a provisional disjoint H&E mask and export a comparison JPEG with legend; do not assert unverified H&E→MSI registration.
- AC5: Score region interiors, overlaps and boundaries with explicit unknown/noise handling and unequal cluster counts. Test analytical cases, degradation and label/row permutation invariance.
- AC6: Report prediction precision, reconstruction distributions/tails/rare-feature error, latent rank/collapse, chemical clustering and numerical gradient diagnostics. Reuse existing model metrics.
- AC7: Provide reproducible script reports under reports/, plots/, numerical_values/ with strict JSON/CSV, saved seeds and provenance. Synthetic demonstrations must be labelled as such.
- AC8: Provide thin executed part_0_* notebooks in this folder, consuming the same report artifacts without model inference.
- AC9: Earlier CPU reference verification is retained as history. After the explicit CUDA-only correction, run CUDA numerical tests and GPU-only model/clustering comparisons using cu118; do not rerun CPU references. Preserve tolerances and report actual commands/results and limits.
- AC10: Preserve existing models/configuration/population/seeds and the user's modified rare-classes notebook. No separate agent branch yet.

## Fixed assumptions

The manually digitized annotation is provisional and derived from the expert illustration; it is not new pathology ground truth. H&E image pixel identity is zero-based row-major index y*width+x. MSI identity is (dataset_id,image_id,source_pixel_id), independent of array row position. Unannotated and unresolved overlap pixels are unavailable. External labels may be used for scoring/matching only; no SVM or other classifier is fitted on external labels. Three synthetic analysis seeds test the verifier, not three trained models. Biological metric tolerances/registration acceptance require independent review; synthetic arithmetic uses analytical oracles.

## Extension: spectral spatial comparison

- EX1: Compare CUDA K-means, DBSCAN and explicit diagonal-covariance GMM on identical synthetic spectra, directly and through existing trained encoders. No artificial latent generated from ground-truth labels in this comparison.
- EX2: Include two stripes, multiple stripes and concentric disk/ring regions. Use prespecified class counts, input noise levels and at least three input seeds; retain actual source spectra and labels.
- EX3: Evaluate three existing clean B0 repetitions and three existing unfrozen-head pretraining repetitions. No training or checkpoint modification. Preserve checkpoint hashes, mass axis, TIC input contract and numerical checks.
- EX4: Produce side-by-side method JPEGs, error maps, spectral profiles, all numerical results and Markdown report. Show first prespecified seeds/repetitions; preserve every scored repetition.
- EX5: Produce source-signal/available-mask JPEGs for all currently selected external reference datasets and embed them in the reference notebook. Label diagnostic-only and normal-only references explicitly; do not invent anatomical masks.
- EX6: Extend and execute part_0_2 as a presentation of the script artifacts; maintain .md and plots as canonical outputs. Test reference geometry, spectral reproducibility, input invariants, numerical model inference and workflow behavior.

## CUDA-only correction

The latest user instruction requires GPU model checks and clustering. No CPU numerical/model references are rerun. Synthetic spectral arithmetic, encoder inference, clustering and small latent-rank computation use CUDA float32. HDBSCAN from the earlier CPU run is archived; the current GPU density method is explicitly named DBSCAN. Preserve the original numerical tolerances and report actual GPU repeatability differences. Host file I/O, plotting and scoring of exported integer assignments are not neural-model or clustering execution.
