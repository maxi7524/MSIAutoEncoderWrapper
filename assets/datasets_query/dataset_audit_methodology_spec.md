# Dataset selection audit — implementation specification

Status: **specification only, nothing here is implemented in the library.** This
document consolidates operations that were carried out ad hoc, as notebook-local
pandas/regex code, across three review notebooks:

- `brain_dataset.ipynb`
- `kidney_dataset_repaired.ipynb`
- `liver_dataset_repaired.ipynb`

All three notebooks use only the existing public interface
(`msi_dataset_manager.exploration.DatasetExplorer`); none of the operations below
exist in `packages/msi_dataset_manager` today. This document is a handoff for a
future implementation session — it describes *what* was done and *why*, with
concrete worked examples and known failure modes, so the logic can be
re-implemented as reusable library functionality (tests, module placement, and API
design are left to that session).

**Explicitly out of scope for this spec:** unifying the m/z axis across organs
(coverage function C(m), candidate-range comparison — see
`brain_dataset.ipynb` section 5). That problem will be solved separately later
and is not part of what should be implemented from this document. Raw-signal
analysis (TIC fraction retained per m/z range, tissue-vs-background ratio) is
also out of scope — it requires per-pixel spectral parsing from raw imzML,
which none of the three notebooks perform.

## 1. Objective technical-duplicate detection

### Problem

Dataset *names* on METASPACE are an unreliable signal for pseudoreplication.
The same raw acquisition is frequently re-submitted or re-processed under a
different name (different mass-tolerance setting, a different analysis
pipeline suffix, a deliberate calibration test) and ends up as a second,
distinct `dataset_id` in the catalogue. Conversely, two *different* physical
samples can carry an identical or near-identical name (observed for
`230618_SIMA9_norm_20_1` in the brain catalogue, submitted twice with
different `pixel_count`).

### Signal

A dataset's METASPACE metadata includes `pixel_count`, `mz_min`, and `mz_max`
(the latter two sourced from the `IMZML_METADATA` diagnostic — i.e. the real
observed range of the raw imzML file, not an annotation-derived range). Two
records sharing an **identical `pixel_count`** and **identical `mz_min`/`mz_max`
rounded to 3 decimal places** almost certainly originate from the same raw
acquisition: independent acquisitions essentially never collide on an integer
pixel grid size *and* both mass bounds simultaneously.

```python
df["mz_min_r"] = df["mz_min"].round(3)
df["mz_max_r"] = df["mz_max"].round(3)
df["cluster_id"] = df.groupby(["pixel_count", "mz_min_r", "mz_max_r"]).ngroup()
cluster_size = df.groupby("cluster_id")["dataset_id"].transform("count")
df["is_duplicate_cluster"] = cluster_size > 1
```

### Refinement: two confidence tiers (required — do not skip)

Applying the raw signal alone across kidney and liver surfaced a genuine false
positive: five liver datasets
(`2022-04-1*_ME_DKFZACLY_S{1,2,3}_W{4,8}_DANneg_..._100x100_100-400_NCE25`)
share `pixel_count=10000` and the same rounded m/z bounds, but differ by a
slide/well identifier (`S1/S2/S3` × `W4/W8`) — five different physical
samples measured under one fixed acquisition template (same grid size, same
nominal mass window), not the same scan resubmitted. A lab that always uses a
standardized protocol will produce exactly this collision.

The fix: within each cluster, strip a defined set of *known* technical/
reprocessing tokens from every member's name and compare what remains
(the "residual"):

```python
def strip_technical_tokens(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"^\d{4}-\d{2}-\d{2}[_ ]", "", s)      # leading YYYY-MM-DD date
    s = re.sub(r"^\d{8}_+", "", s)                     # leading YYYYMMDD_
    s = re.sub(r"[-_]?\d+ ?ppm\b", "", s)               # mass-tolerance suffix
    s = re.sub(r"\btic\b", "", s)                       # "total ion count" shorthand
    s = re.sub(r"_(?:aq_ml|aq|ml)$", "", s)             # pipeline suffixes
    s = re.sub(r"-total ion count$", "", s)
    s = re.sub(r" - root mean square$", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s
```

- **`high_confidence_duplicate`** — every member's residual is identical →
  safe to keep one, exclude the rest.
- **`ambiguous_shared_template`** — residuals differ → do **not**
  auto-exclude; the difference (e.g. `w8` vs `w4`) is presumptively a real
  sample identifier, not a technical artifact. Surface for manual review.

### Keeper selection (within a high-confidence cluster)

Prefer the member whose name does **not** carry a recognized technical
suffix (`_ML`, a ppm number, `-total ion count`); break remaining ties by
`dataset_id` ascending, for determinism:

```python
TECH_SUFFIX_PENALTY = re.compile(r"(?:_ml$|_v$|ppm$|-total ion count$)", re.IGNORECASE)
```

### Known limitations (report, do not silently paper over)

- **Rounding-boundary false negatives.** In the same DKFZACLY family, a sixth
  member (`..._S1_W4_...`) has `mz_max` differing from the other five by
  ~0.0008 — the *fourth* decimal — which crosses the 3-decimal rounding
  boundary and keeps it out of the cluster entirely, even though it is
  presumably part of the same protocol family. The rounding precision is a
  tunable parameter with real trade-offs in both directions, not a solved
  constant.
- **The residual-convergence check is itself a heuristic list of tokens.** It
  will not catch reprocessing conventions it hasn't seen (e.g. the liver
  `id2_calnaf_quadraticen` / `2026_03_25_calnaf_quadraticen` pair, and the
  `2026_02_24_Rep-01` / `2026_02_19_Rep-01` pair, both stayed
  `ambiguous_shared_template` in this run — plausibly true duplicates, but the
  token list wasn't confident enough to say so). Extending the token list is
  expected maintenance, not a one-time task.

### Regression fixtures worth preserving

| Organ | Cluster | Verdict |
|---|---|---|
| brain | `PNNL05A_V6b_CLMCAFAMM_Lipids_885_{3,5,10}ppm` | high-confidence (explicit example the user supplied) |
| brain | `*_AQ` / `*_AQ_ML` pairs (7 pairs, same day's acquisitions) | high-confidence |
| brain | `230618_SIMA9_norm_20_1` (two different `dataset_id`s, different `pixel_count`) | **not** a duplicate — correctly not clustered |
| kidney | (none — kidney's 60-dataset broad pool has zero exact clusters) | — |
| liver | `liver storage day 2 tic` / `-110ppm` / `50ppm` / duplicate literal name (4-way) | high-confidence |
| liver | `liver rn lip tic` / `-50ppm` | high-confidence |
| liver | `liver 2-3 50um ms range 300_1500 213_306 pNA-` (literal duplicate submission, same name and stats) | high-confidence |
| liver | DKFZACLY `S{1,2,3}_W{4,8}` family (5 members) | **ambiguous — do not exclude** (false-positive test case) |
| liver | `2026_02_24_Rep-01` / `2026_02_19_Rep-01` | ambiguous (needs a smarter token rule to resolve) |

## 2. Named calibration / QC mass-shift variant

Two independent, organ-specific examples of the same pattern:

- brain: `granular_layer_mouse_brain` vs
  `granular_layer_mouse_brain_null_mz_shift_10_from_2575` — identical
  `pixel_count` and `mz_min`, `mz_max` offset by **exactly 10**.
- kidney: `kidney_test_metabolites` vs
  `kidney_test_metabolites_null_mz_shift_10_til_550` — identical
  `pixel_count` and `mz_max`, `mz_min` offset by **exactly 10**.

Both were detected in the notebooks by a literal name check
(`name.str.contains("null_mz_shift")`), which only works because both labs
happened to name the variant descriptively. **A more general detector should
not rely on the name**: the underlying signature is *same `pixel_count`, one
of `{mz_min, mz_max}` identical, the other differing by a small integer*
(observed: exactly 10 in both cases). A general implementation should search
for that numeric signature directly and treat the name match as corroborating
evidence, not the primary signal.

The kidney case is the more consequential finding: the QC-shifted variant is
currently part of the **already-downloaded** 30-dataset kidney corpus (see
`kidney_dataset_repaired.ipynb` section 4) — it was not caught by the earlier
manual exclusion pass.

## 3. Morphology / anatomical-region heuristic tagging

Per-organ keyword regexes, checked case-insensitively against `name`, used to
flag likely regional/microanatomical fragments as opposed to whole-organ
sections:

| Organ | Keywords used | Hits (out of broad pool) |
|---|---|---|
| brain | `purkinje`, `granular_layer`, `molecular_layer`, `fibers_layer`, `_hip_`/`_hip$`, `_cer_`/`_cer$`, `cerebellum`, `hippocamp`, `striatum`, `olfactory`, `midbrain`, `substantia`, `hypothalamus`, `thalamus`, `cortex` (excluding `...coronal` context) | 15 / 223 |
| kidney | `cortex`, `medulla`, `papilla`, `pelvis`, `calyx`, `glomerul*` | 2 / 60 |
| liver | `lobe`, `periportal`, `pericentral`, `zonation`, `capsule`, `portal` | 0 / 82 |

**This tier is advisory only and must never be auto-excluded**, except for
cases a domain expert names explicitly (brain's four cerebellar-layer
datasets were named directly by the user and were the only morphology-based
exclusions applied in `brain_dataset.ipynb`). The keyword lists above are a
reasonable seed, not a validated taxonomy — liver's zero hits, in particular,
only mean "no keyword matched," not "no liver dataset is a regional
fragment."

Suggested config shape for an implementation: a per-organ (or
per-`organism_part`) mapping of regex patterns, extensible without touching
call sites.

## 4. Low-pixel-count / likely-test-scan flag

A fixed absolute threshold does **not** transfer across organs. Brain's pool
contains genuine calibration/test scans down to 25 pixels, justifying a
`<500` cutoff there. Kidney's pool has a minimum of 5025 pixels — the `<500`
cutoff would find nothing and is meaningless. Liver's pool is systematically
much smaller (many IR-MALDESI `Lipids*`/`Tissue*` acquisitions sit at
500–1000 pixels **by design**, and are already part of the accepted 18-dataset
corpus) — reusing brain's `500` would incorrectly flag legitimate, already
-approved liver datasets.

Threshold used per organ in this audit (informal, eyeballed against each
organ's own `pixel_count.describe()`):

| Organ | Threshold | Rationale |
|---|---|---|
| brain | `< 500` | pool contains scans down to 25 px |
| kidney | *(not applied — nothing to flag)* | pool minimum is 5025 px |
| liver | `< 200` | pool 5th percentile is ~465 px; a `500` cutoff would flag accepted datasets |

An implementation should derive the threshold from each queried pool's own
distribution (e.g. a low percentile, or an explicit per-organ config value)
rather than hard-coding one number.

## 5. Biological-series / name-family grouping

From `brain_dataset.ipynb` only (not re-run for kidney/liver in this pass,
but the same token-stripping approach applies). Purpose: identify datasets
that are technical replicates or sequential sections of the *same*
biological source, so they can be constrained to the same train/validation/
test split instead of leaking across splits.

```python
def biological_series_key(name: str) -> str:
    s = str(name).lower()
    s = re.sub(r"^\d{4}-\d{2}-\d{2}[_ ]", "", s)
    s = re.sub(r"^\d{8}_+", "", s)
    s = re.sub(r"_(?:aq_ml|aq|ml)$", "", s)
    s = re.sub(r"_\d+ppm$", "", s)
    s = re.sub(r"-total ion count$", "", s)
    s = re.sub(r" - root mean square$", "", s)
    s = re.sub(r"_replicate\d+$", "", s)
    s = re.sub(r"_s\d+$", "", s)
    s = re.sub(r"_\d+$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s
```

This is deliberately conservative (only strips recognized technical tokens)
so it does not silently merge different animals under one label. It groups,
it does not pick a representative — which record to keep per series is left
to manual review.

## 6. Audit-diff against an existing curated selection

For a dataset with an existing `filter.json`/`selection.json`
(`exclude_dataset_ids`, `dataset_ids`), intersect the newly-found objective
exclusions (§1 high-confidence duplicates ∪ §2 QC-shift variants) against
both sets to answer two questions in one step:

- **Missed by the earlier manual review** — new finding ∩ currently
  *selected* IDs (kidney: 1 hit — the QC-shift variant is in the downloaded
  30).
- **Already caught** — new finding ∩ already-*excluded* IDs (kidney: 0;
  liver has never had a manual exclusion pass, so this is trivially 0).

This diff is what makes the audit actionable rather than purely descriptive.

## 7. Suggested library surface (non-binding sketch)

Not decided, offered only as a starting point for the implementing session.
Natural home: a new module alongside `msi_dataset_manager.exploration`
(e.g. `msi_dataset_manager.exploration.audit`), operating on the
`pandas.DataFrame` returned by `DatasetExplorer.filter()`/`.search()` so it
composes with the existing notebook workflow rather than replacing it:

- `detect_technical_duplicates(df, rounding=3, technical_tokens=...) -> DataFrame`
  — adds `cluster_id`, `cluster_confidence`, `duplicate_excluded` columns per
  §1, generalizing the current ad hoc token list into a configurable
  parameter.
- `detect_mz_shift_variants(df, pixel_count_col=..., tolerance=...) -> DataFrame`
  — generalizes §2 beyond name matching to the numeric signature.
- `flag_regional_fragments(df, keywords: dict[str, list[str]]) -> DataFrame`
  — per-organ keyword tagging per §3, keyed by `organism_part`.
- `flag_low_pixel_count(df, threshold: float | None = None, percentile: float | None = None) -> DataFrame`
  — per §4, threshold derived from the queried pool if not given explicitly.
- `biological_series_key(name: str) -> str` — per §5, or a
  `DatasetExplorer`-level `.assign_series_ids()` convenience.
- `diff_against_selection(new_exclusions, filter_json_path, selection_json_path) -> dict`
  — per §6.

Whether these belong on `DatasetExplorer` itself, as free functions, or as a
small `DatasetAudit` companion class operating on its result table is an
implementation decision, not something resolved here.

## 8. What this spec does not cover

- Cross-organ m/z range unification (`brain_dataset.ipynb` section 5) —
  explicitly deferred by the user to a separate effort.
- Raw-signal TIC-fraction / tissue-vs-background analysis
  (`brain_dataset.ipynb`, "Etapy 4–5") — requires per-pixel imzML parsing,
  not attempted in any of the three notebooks.
- Any actual code change to `packages/msi_dataset_manager` or
  `src/msi_autoencoder_wrapper` — none was made; everything above lived as
  notebook-local pandas/regex code operating on `DatasetExplorer` output.
