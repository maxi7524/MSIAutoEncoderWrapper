# Run a catalog-driven autoencoder analysis

This guide runs an existing notebook collection whose `analysis_settings.yaml` declares a common precompute strategy.

## Scope

The command resolves campaign manifests, selects models from the YAML catalog, writes CSV inputs, and then the notebooks render figures from those files. It does not generate notebooks.

## Execution

### Run the configured strategy

Run from the repository root.

```bash
SETTINGS=assets/experiments/autoencoder_architecture/notebooks/segmentation_model/17_09_26_metaspace_heads_vpu_contractive/analysis_settings.yaml
MPLCONFIGDIR=/tmp/msi-matplotlib UV_CACHE_DIR=/tmp/msi-uv-cache \
uv run --extra cu118 python -m msi_autoencoder_wrapper.analysis.precompute \
  --settings "$SETTINGS"
```

Use `--dry-run` to validate sources, technical model selectors, aliases, output paths, and strategy order without loading models.

```bash
uv run --extra cu118 python -m msi_autoencoder_wrapper.analysis.precompute \
  --settings "$SETTINGS" --dry-run
```

The full command requires CUDA for inference. `campaign_training_dynamics` is CPU-only because it reads manifests and histories only.

### Execute the notebook collection

After precompute succeeds, execute notebooks in numeric order. This records figures and exposes stale CSV contracts immediately.

```bash
for NOTEBOOK in assets/experiments/autoencoder_architecture/notebooks/segmentation_model/17_09_26_metaspace_heads_vpu_contractive/part_*.ipynb; do
  MPLCONFIGDIR=/tmp/msi-matplotlib UV_CACHE_DIR=/tmp/msi-uv-cache \
  uv run --extra cu118 jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=900 "$NOTEBOOK"
done
```

See [analysis internals](../../library-internals/metrics-analysis-and-visualization/catalog-driven-precompute.md) for artifact flow and [analysis development](../../developer-guides/analysis/catalog-driven-precompute.md) for adding a strategy.

## File configuration

### YAML model catalog

`models` maps a stable analysis alias to one technical selector. A selector must identify exactly one model for every selected repetition; use `source` and `grid_id` when labels are reused across campaigns. `display_label`, `tags`, and `visualization` are presentation metadata, not selectors.

```yaml
models:
  vpu_contractive:
    display_label: VPU + contractive
    select:
      source: vpu_contractive_interaction_expanded
      grid_id: grid_0001
    tags:
      contractive: true
    visualization:
      order: 1
      color: "#D55E00"
      line_style: dashed
```

`groups` defines ordered alias sets for an analysis. A local two-model comparison uses `selected_models`; global stages use the resolved catalog.

### Read the generated inputs

The common control artifacts are stored in `precompute/` beside the settings file:

- `resolved_model_catalog.csv` records the exact selected repetitions and technical source identities.
- `visualization_contract.csv` records labels, ordering, colours, and line styles.
- `execution_plan.json` records the resolved strategy stages.

Each stage writes its own `part_*_results/` directory. Do not load model checkpoints in a notebook; use the result CSV files and the notebook helpers `load_model_catalog`, `load_visualization_theme`, and `select_catalog_frame`.
