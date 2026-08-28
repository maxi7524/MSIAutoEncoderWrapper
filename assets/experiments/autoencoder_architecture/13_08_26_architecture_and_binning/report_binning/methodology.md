# Binning / Inverse-Binning Analysis — Methodology

Status: DRAFT for review. Items marked `#TODO` are open decisions or things that need
verification against the codebase before/while implementing. Nothing described here has
been implemented yet.

## 1. Objects and notation

- $X = \{(m_i, I_i)\}_{i=1}^n$ — the original sparse spectrum, native $m/z$ coordinates
  (read via `MSIBaseReader.GetSpectrum` / `PixelDataset.get_raw_item`, **not** through
  `PixelDataset.__getitem__`, which already returns the binned+normalized spectrum).
- $\mathrm{B}(X) = \{(c_j, J_j)\}_{j=1}^d$ — spectrum after forward binning onto a fixed,
  dense grid (`LinearBinning`, step $\Delta m$).
- $\mathrm{INB}(X) := \mathrm{INB}(\mathrm{B}(X)) = \{(\hat m_k, \hat I_k)\}_{k=1}^q$ —
  spectrum after applying an inverse binner to $\mathrm{B}(X)$.

All three objects carry explicit $m/z$ coordinates. Comparisons are always done by
**point matching within a tolerance**, never by projecting onto a shared axis or by
linear interpolation. This applies uniformly to every comparison below.

### Three comparisons — never conflated, never combined into one plot/table

1. `forward`: $\mathrm{B}(X)$ vs $X$ — error introduced by forward binning alone
   (axis quantization, in-bin summation).
2. `inverse_binned`: $\mathrm{INB}(X)$ vs $\mathrm{B}(X)$ — error introduced by the
   inverse binner's compression alone. Computed directly on native $m/z$ coordinates of
   both sides. **Not** $\mathrm{B}(X)$ vs $\mathrm{B}(\mathrm{INB}(X))$ — re-binning the
   inverse-binner output would erase exactly the localization error we want to measure.
3. `inverse_original` (round trip): $\mathrm{INB}(X)$ vs $X$ — end-to-end write/read
   error.

$d_X(X,\mathrm{INB}(X)) \le d_X(X,\mathrm{B}(X)) + d_X(\mathrm{B}(X),\mathrm{INB}(X))$ is
an orientation check, not something we compute as a derived metric — each comparison is
reported on its own.

## 2. Matching

Reuse `metrics.spectral_points.match_spectral_points(reference_mz, reference_intensity,
candidate_mz, candidate_intensity, tolerance, tolerance_unit, matching_strategy)` as-is —
already implements `Da`/`ppm` tolerance and `nearest` / `one_to_one` / `local_mass`
matching strategies. No new matching code planned.

**Resolved**: `one_to_one` is the default for every quantitative metric in the new
topic modules (matches by $m/z$, each point used at most once — this is the actual
comparison we want). For reference: `nearest` looks up the closest candidate within
tolerance for every reference point independently (a candidate can be reused — diagnostic
only, never used for reported numbers). `local_mass` also looks up candidates within
tolerance per reference point, but instead of picking one it **sums the intensity of every
candidate found in that tolerance window** into that single reference point's matched
value — this models "several fine points collapse onto one coarse point" (e.g. many raw
peaks inside one forward bin). Existing `analysis/autoencoder/binning/analysis.py`
already defaults to `local_mass` everywhere; the new topic modules use `one_to_one`
instead, deliberately — the two are not swapped or mixed for the same reported number.

## 3. Metrics

Reuse `metrics.spectral_points.spectral_point_metrics(...)` — **already implemented**,
already returns per matched-pair comparison: `localization_mae/rmse_da/ppm`,
`localization_median/q90/q95/q99/max_da/ppm`, `peak_recall`, `peak_precision`,
`matched_intensity_fraction`, `local_intensity_relative_l1`, `tic_relative_error`,
`size_ratio`, `size_reduction`, `wasserstein`, `cosine_similarity`, `spectral_angle`.
This covers essentially the whole originally-planned metric set. Nothing here gets
reimplemented.

Two metrics are genuinely missing and need to be added (validated against the
definitions below, unit-tested analytically):

- **$C_{\mathrm{merge}}$ — peak collision rate** (`metrics.spectral_points.peak_collision_rate`,
  implemented and smoke-tested). Fraction of candidate points fed by ≥2 reference peaks
  whose mutual span exceeds one tolerance width — i.e. peaks that would not have merged
  had the candidate resolved them individually. "Significant contribution" is an
  optional `min_relative_height` cutoff (fraction of the tallest contributor; default 0,
  no filtering). Captures irreversible loss of resolution even when localization error
  is small.
- **$E_{\mathrm{unmatched}}$ — intensity of unmatched candidate points**, relative to
  total candidate intensity. Complements `peak_recall`/`peak_precision` (which are
  point-count based) with an intensity-weighted view of "how much of what we produced
  wasn't asked for."
- **Signal-retention-by-quantile curve** (resolved from your note — a scalar ratio hides
  *whether* the lost mass was noise or real peaks). Rank reference points by intensity,
  descending. For $q \in \{0.50, 0.90, 0.95, 0.99\}$, find $k_q$ = the smallest number of
  top-ranked reference points whose cumulative intensity reaches fraction $q$ of total
  reference intensity, then report `recall_at_quantile(q)` = fraction of those top-$k_q$
  points that are matched. Low recall at $q{=}0.5$ (the few points carrying half the
  signal) means real peaks are being dropped; unmatched mass showing up only once $q$
  approaches 0.99 means it is thin/noise-like. Reported per (method, parameter,
  normalization) alongside $E_{\mathrm{unmatched}}$, not instead of it.

Existing inverse binner implementations (`ThresholdInverseBinner`,
`CumulativeMassInverseBinner`, `PeakRegionInverseBinner`) will be reviewed for
correctness against their own docstrings/spec before being trusted in any sweep —
this is a validation pass, not a rewrite, unless a real bug is found.
**Resolved — diagnostics is opt-in.** `last_diagnostics`/`get_last_diagnostics()` is the
per-call dict already built by `Threshold`/`CumulativeMass`/`PeakRegion` inverse binners
(input/valid/selected bin counts, retained mass fraction, compression ratio, etc.) —
cheap per call, but adds up when a sweep calls `__call__` thousands of times per
parameter point. All four inverse binners get a constructor parameter
`track_diagnostics: bool = False`: when `False` (default), `__call__` skips building the
diagnostics dict entirely and `last_diagnostics` stays `{}`; when `True`, behaves exactly
as today. `TopPeaksInverseBinner` gets `last_diagnostics`/`get_last_diagnostics()` added
under the same toggle (currently missing entirely). No rewrite into a
`PeakRegionInverseBinner` wrapper — its selection logic (greedy over sorted intensity,
window expansion, early stop at `max_bins`) is genuinely different from peak-region
detection, not an equivalent special case.

### Normalization variants — always computed together, never mixed in one plot

Every metric above is computed three times, on differently normalized intensities:

- `raw` — untouched intensities.
- `tic` — $\tilde I = I / (\sum I + \varepsilon)$.
- `max` — $\tilde I = I / (\max I + \varepsilon)$.

`tic_relative_error` is only meaningful on `raw` (it is ~0 by construction after
normalization) — reported under `raw` only, flagged as N/A elsewhere.

Normalization is **not** the training-time `PixelDataset(normalization=...)` setting.
It is applied post-hoc, per metric call, on top of raw cached intensities — the
precompute step always caches raw, unnormalized $(X, \mathrm{B}(X), \mathrm{INB}(X))$
triples, and normalization is a parameter of the metric layer.

Both Part 1 (parameter fitting) and Part 2 (method comparison) are run once per
normalization and results are compared side by side ("does method A degrade more under
`max` normalization than method B") — this is itself one of the analyses, not just
repeated bookkeeping.

**Confirmed scope**: behavior is characterized jointly across three axes — different
inverse-binner methods, different $m/z$ sub-ranges, different normalizations — and for
every combination of the three we want to determine the optimal strategy. This is why
Part 1 (§6) is structured as method × region × normalization, not a single sweep with
normalization as an afterthought.

## 4. Mass axis range

`PyImzMLReader.GetXMin/GetXMax/GetXAxis` currently read spectrum index 0 only, not a
true dataset-wide range — wrong for centroid/variable-length data where different
pixels can have different local $m/z$ extents. **Resolved**: `m2aia` (native reader,
faster, true dataset-wide `GetXAxis()`) is not installed in this environment
(`import m2aia` fails) — checked directly, not assumed. Default reader for this work
stays `PyImzMLReader`; the true global min/max is computed by precompute itself, scanning
every sampled spectrum's native axis, rather than trusting any reader method — this makes
the fix reader-agnostic, so switching to `M2aiaReader` later (once available) needs no
changes downstream. Switching the *default* reader to `M2aiaReader` for speed once it's
installed is a separate, later decision, not blocking this work.

## 5. Sampling

- Parameter-fitting sweeps (Part 1) run on a **seeded random sample** of spectra,
  default $N = 10{,}000$, both $N$ and seed exposed as notebook-top variables.
  `C57BL6_Kidney_02-pos` has 5959 spectra total — below the default sample size, so in
  practice the whole image is used for this dataset; the sample cap matters for larger
  datasets swapped in later.
- Method comparison at optimal settings (Part 2) and any full-image diagnostic
  (spatial error maps, ion-image comparison) runs on **all** pixels of the current
  image, with interactive plots for browsing per-pixel results (not static grids of a
  few hand-picked spectra).
- Sub-region analysis (§6) reuses the **same single global sample** — it never redraws.
  Restricting to a sub-range only crops the $m/z$ axis of already-sampled spectra before
  re-binning/re-inverse-binning them on that window (see §6 step 3); the set of spectra
  themselves does not change.

## 6. Plan of analyses

Part 1 — parameter fitting (per method, per normalization); establishes what "optimal"
means before any cross-method comparison happens.

1. **Forward binning $\Delta m$ sweep — decided once, globally, first**, because it
   determines every downstream input. Grid tested only toward *larger* $\Delta m$ than
   the dataset's native resolution (finer than native can't be recovered anyway):
   $\Delta m \in \{0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.020\}$ (grid itself is a
   notebook-top variable). For every metric separately (different scales — never on a
   shared axis), plotted as a **distribution** (histogram/ECDF across sampled spectra),
   with each $\Delta m$ as its own colored series on the same plot — i.e. "for this
   density/step, what's the shape of the error distribution across the image," not just
   a single summary statistic per $\Delta m$.
2. **Inverse binner parameter sweep**, per method, at the $\Delta m$ chosen in step 1.
   Guided primarily by Masserstein/$W_1$ distance as the localization proxy (per your
   note — it best captures whether the spectrum "came apart"), with the rest of the
   metric set reported alongside as context, never collapsed into one score. A
   **manual parameter slot is exposed at the top of the notebook** — you pick the final
   parameter set per method by inspection of the sweep plots; nothing is auto-selected.
   For every metric, also plot the **3 best and 3 worst spectra** (by that metric's own
   ranking) — the point is to see, per metric, which kind of error it actually weighs
   (a metric that never flags the same spectra as another is measuring something
   different, not a redundant restatement). Reuse the existing mirrored comparison plot
   (`visualization.spectra.plot_sparse_spectrum_match` — reference intensity plotted
   positive, candidate plotted negative/mirrored, matched pairs connected, unmatched
   points marked, residual panel showing `matched_reference − matched_candidate` per
   pair) rather than inventing a new plot style — it already does exactly this. The
   parameter choice stays global (one decision per method), these plots are for
   understanding failure modes, not for selecting a different parameter per spectrum.
   No cap on plot count as long as each one is informative; drop a plot only if it
   turns out uninformative in practice, not pre-emptively.
3. **Sub-range dependency analysis**, repeating steps 1 and 2 but **recalibrated on
   restricted $m/z$ windows** (default width 100, e.g. 300–400, 400–500, …; both window
   width and boundaries are notebook-top variables) — a genuinely separate binning run
   per window, not the global run merely filtered/evaluated on a sub-range. Goal:
   characterize how the optimal parameter value depends on $m/z$ location, for both
   $\Delta m$ and inverse-binner parameters. Global (whole-range) and local (per-window)
   behavior are always reported together, not as a replacement for one another.

Part 2 — method comparison at optimal settings (uses the parameter choices/grids from
Part 1, including per-region optimal settings from step 3):

4. Cross-method comparison using each method's own optimal settings (global and, where
   relevant, per-region), across all metrics and all three normalizations, repeated for
   both `inverse_binned` and `inverse_original` comparisons.

Supporting analyses, run at optimal settings once Part 1/2 give a concrete config to
plug in (order matches the original request):

5. Per-spectrum breakdown: best/median/worst spectra per metric (`metric_order`),
   grouped by TIC / peak count / max intensity / dominant-peak share / collision count —
   separates transformation error from spectrum-complexity effects.
6. $m/z$-localization profile: error vs $m/z$ position, both raw and normalized, ppm and
   Da.
7. Spatial maps: per-pixel metric values projected back onto image coordinates
   (`reader.MapSpectrumValuesToImage`) — interactive, browsable, not static thumbnails.
8. Ion-image comparison for selected $m/z$/tolerance: **three** images side by side —
   original $X$, binned $\mathrm{B}(X)$, and round-trip $\mathrm{INB}(X)$ — not just
   original vs round-trip, so it's visible *where* signal is lost at each stage.
   Relative $L_1$, spatial correlation, optionally SSIM computed pairwise between all
   three. Each image browsable (pick $m/z$ interactively), not a fixed static set.

## 7. Code organization

Organized **by analysis topic**, not by architectural layer — a layer-based split
(metrics.py / views.py / overviews.py all under one generic pattern) would just
re-wrap the same global calls under an empty abstraction. Structure:

```
analysis/autoencoder/binning/
    precompute.py                  # single shared precompute step (see below)
    binner_forward_analysis.py     # §6 step 1
    inverse_binner_analysis.py     # §6 step 2
    region_dependency_analysis.py  # §6 step 3
    method_comparison_analysis.py  # §6 step 4
    per_spectrum_analysis.py       # §6 step 5
    localization_analysis.py       # §6 step 6
    spatial_analysis.py            # §6 step 7
    ion_image_analysis.py          # §6 step 8
```

Kept at the existing location (`analysis/autoencoder/binning/`), adapted to current
conventions, nothing else in the surrounding architecture touched.

Each topic file exposes a small number of functions that take the precompute object
(plus config: parameter grid, normalization, region bounds, sample) and internally call
the **global** `metrics` (`metrics/spectral_points.py` + the two new metrics) and
**global** `visualization` (theme-driven plotting primitives) modules, returning
figures/tables ready to drop into a notebook cell and still modifiable afterward (no
plotting logic lives inside `metrics`; no metric math lives inside a topic file).

`precompute.py` is global/shared across all topic files: for the sampled spectrum set,
computes and caches raw $X$, $\mathrm{B}(X)$ under the chosen $\Delta m$(s), and
$\mathrm{INB}(X)$ for every (method × parameter-grid-point) combination needed by the
sweeps — this is the expensive step, run once; every topic-analysis function then reads
from these cached arrays instead of recomputing binning per metric call.

## 8. Notebook

New notebook created **alongside** the two existing ones in
`assets/notebooks/tests/05_08_26_binning_analysis/` (not overwriting them) — file name
`#TODO` (suggest `optimal_binning_analysis.ipynb` or similar, pick on review). All
sweep grids ($\Delta m$ set, inverse-binner parameter grids, sample size/seed, region
window width/boundaries) are variables in one of the first cells.

## 9. Tests

Synthetic cases as originally listed (forward binning edge cases, threshold /
cumulative-mass / peak-region edge cases, matching edge cases, metric correctness
including the two new metrics) — written once the corresponding code exists, not before.
