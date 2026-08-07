---
name: binning-analysis-conventions
description: Project-specific conventions for the binning/inverse-binning analysis modules under analysis/autoencoder/binning/ (precompute.py, binner_forward_analysis.py, inverse_binner_analysis.py, and future topic modules for this domain) — docstring depth, logging/progress, and plot color/grouping rules. Load this whenever adding or editing a topic-analysis module in that package, in addition to the global engineering-standards skill.
---

# Binning-analysis module conventions

This package is organized **by analysis topic**, one file per analytical question
(`binner_forward_analysis.py`, `inverse_binner_analysis.py`, and future
`region_dependency_analysis.py` / `method_comparison_analysis.py` / etc.), not by
architectural layer. Each topic file's public functions are called directly from a
notebook. These rules exist so the notebook itself can stay thin — the notebook should
not need to restate what a function does, because the function's docstring already
says it precisely.

Always apply the global `engineering-standards` skill first (hierarchical `#`/`##`/`###`
comments, `get_custom_logger`, Sphinx docstrings, lazy `%s` logging). This skill adds
domain-specific requirements on top of it.

## Docstrings must state the full methodology, not just the signature

For every public function in a topic module, the docstring must answer, precisely
enough that the notebook cell calling it needs no extra prose:

- What is being compared (which two representations — e.g. `B(X)` vs `X`, or a
  swept-parameter reference vs candidate) and in which direction.
- What the record/output is grouped or keyed by (e.g. "one row per (spectrum, delta_m,
  normalization, metric)").
- For plotting functions: what visual channel encodes what (color = group, linestyle =
  comparison, x position = ...), and what NOT to read into the plot (e.g. "points from
  different methods are not connected — their parameter spaces are not comparable").
- Any invariant carried over from `methodology.md` that a reader would otherwise have
  to go re-derive (e.g. "matching is position-only, independent of normalization").

A one-line docstring is not acceptable for anything doing matching, aggregation, or
plotting in this package — only for trivial pass-through helpers.

## Logging and progress

- Every topic module gets a module-level `logger = get_custom_logger(__name__)`.
- `BinningPrecompute` and any function that loops over the full spectrum sample must
  report progress — this sample can be thousands of spectra and a single sweep can take
  from minutes to hours (`match_spectral_points` is an O(points) Python loop per
  spectrum, not vectorized). Use `tqdm.auto.tqdm` for the progress bar (works in both
  notebook and terminal) and `logger.info` for the start/end of each major stage
  (sample size, delta_m/parameter grid point being processed, cache hit vs recompute).
- Log at `debug` the resolved config of every binner/inverse-binner built during a
  sweep (`get_config()` is already JSON-safe — log it directly).

## Plotting: color and grouping conventions

- **Continuous swept parameters** (e.g. `delta_m`, a quantile, a retained fraction)
  get colors from a **sequential colormap** sampled across the sweep's index range
  (e.g. `theme.image_colormap`, default `viridis`), never from `theme.model_palette`
  cycling — the palette has 6 entries and silently wraps around (two different swept
  values end up the same color) for any grid larger than 6, which is common.
- **Categorical groups that are not orderable** (e.g. different inverse-binner
  *methods*, as opposed to one method's own parameter grid) get colors from
  `theme.model_palette`/`theme.color_for_model`, consistently reused across every plot
  in a notebook run (same method → same color everywhere).
- **A second categorical dimension on the same plot** (e.g. `comparison` —
  `inverse_binned` vs `inverse_original`) is encoded as **linestyle/marker**, not a
  second color axis — color stays reserved for the primary group.
- Never let a tradeoff/line plot connect points that belong to different,
  non-comparable parameter grids (e.g. different inverse-binner methods) with one line
  — group by method (or whatever defines a comparable parameter axis) and give each
  group its own line/marker series, even when several groups share one axes object.
- Distribution plots with several overlapping series (e.g. one histogram per swept
  value on one axes) must be visually distinguishable where they overlap — use
  `histtype="step"` outlines or explicit alpha low enough that stacked series remain
  distinguishable, never opaque filled bars for more than ~2 overlaid series.
