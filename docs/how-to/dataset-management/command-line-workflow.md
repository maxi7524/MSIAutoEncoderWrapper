# Use the msi-datasets CLI

```{admonition} METASPACE API access
:class: warning

As of this writing, METASPACE does not allow this project to query or
download datasets through its public API. `query` and `download` below
describe the intended behavior for when API access is available. Until then,
`msi-datasets` is used mainly to manage datasets that are already downloaded:
see [`compose`](#compose-merge-a-cohort-with-molecule-masks) below, and
[Inspect the catalog](inspecting-the-catalog.md).
```

`msi_dataset_manager` installs one command-line entry point, `msi-datasets`,
with three subcommands: `query`, `download`, and `compose`. There is no
separate `merge`, `download-merge`, or `import-local` command — composing a
cohort is the only supported path from local files to a merged, catalog-backed
dataset. This guide is a complete reference for the three subcommands; the
task-oriented guides linked from each section explain the concepts, file
formats, and failure modes in depth.

Every subcommand accepts `--workspace-path` (default: `workspace`, resolved
relative to the current directory) and prints the invocation directory,
resolved workspace path, resolved cohort ID, and target SQLite catalog path
before doing anything else — useful for confirming which catalog a command is
about to write to before it runs. `query` and `download` print the **working
catalog** (`configs/datasets/<cohort_id>/<cohort_id>.sqlite`); `compose`
prints the **composed catalog**
(`datasets/<cohort_id>/<cohort_id>.sqlite`) instead, since it does not use the
working catalog at all. See
[Choose the cohort catalog](downloading-datasets.md#choose-the-cohort-catalog)
for how `<cohort_id>` is resolved when a subcommand has no explicit
`--cohort-id`.

```bash
msi-datasets {query,download,compose} ...
```

## `query`: discover datasets without downloading

```bash
msi-datasets query \
  --workspace-path workspace \
  --source metaspace \
  --filters workspace/configs/datasets/kidney/filter.json \
  --selection workspace/configs/datasets/kidney/selection.json \
  [--cohort-id kidney]
```

Runs `--filters` (a JSON filter object, see
[Discover external datasets](discovering-datasets.md#filter-datasets)) against
the provider, upserts accepted records into the working catalog, and writes
`--selection`. Full detail: [Discover external datasets](discovering-datasets.md)
and [Review and save selections](filtering-and-selection.md).

## `download`: materialize a selection

```bash
msi-datasets download \
  --workspace-path workspace \
  --source metaspace \
  --selection workspace/configs/datasets/kidney/selection.json \
  [--annotation-options workspace/configs/datasets/kidney/annotation_options.json] \
  [--dataset-id DATASET_ID ...] \
  [--profiles workspace/configs/datasets/kidney/profiles.csv] \
  [--manifest workspace/configs/datasets/kidney/materialization.json] \
  [--cohort-id kidney]
```

Downloads (or reuses) every selected dataset's imzML/ibd pair, then retrieves
and writes its annotations as CSV files beside it — **not** into any SQLite
catalog; only file materialization status is recorded in the working catalog.
`--dataset-id` may be repeated to restrict the run to a subset without editing
the selection file. `--profiles` rotates between several provider accounts on
quota errors. Full detail: [Download selected datasets](downloading-datasets.md).

## `compose`: merge a cohort with molecule masks

```bash
msi-datasets compose \
  --workspace-path workspace \
  --cohort-id kidney \
  --source metaspace \
  --selection workspace/configs/datasets/kidney/selection.json \
  [--config workspace/configs/datasets/kidney/composition_config.json] \
  [--dataset-id DATASET_ID ...] \
  [--row-width 128] [--max-fdr 0.1] [--minimum-dataset-occurrence 2] \
  [--unannotated-ratio 1.0] [--unannotated-amount N] [--random-seed 0]
```

Imports each input dataset's annotation CSVs into a new, self-contained
composed catalog, merges every canonical local dataset in the cohort into one
image, and builds a cohort-wide molecule-occurrence index
(`annotation_index.json`) alongside it. Contacts no provider — datasets must
already be local (typically via `download` above). `--cohort-id` is required
for this subcommand — there is no fallback resolution. Full detail:
[Compose a cohort dataset](composing-a-cohort.md).
