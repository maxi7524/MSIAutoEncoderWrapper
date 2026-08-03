# Merge local MSI datasets

Merging writes selected spectra from several source images into one imzML/ibd
pair and records source provenance for every merged index.

## Purpose and available operations

### Spectrum selection

Explicit `spectrum_ids` override automatic selection. Otherwise all annotated
spectra are included and optional unannotated spectra are sampled by ratio or
absolute amount.

### Provenance

The catalog maps each merged spectrum index to source, source dataset ID, and
source spectrum ID. Source annotations remain queryable after source files are
removed.

## Detailed instructions

### Run a configured merge

```bash
.venv/bin/python assets/scripts/datasets/manage_datasets.py merge \
  --workspace-path data/tutorial_workspace \
  --config data/tutorial_workspace/datasets/merge_kidney_pilot_v2.json
```

Configuration paths are resolved relative to the configuration file.

### Control automatic sampling

`unannotated_ratio` requests `floor(annotated_count * ratio)` spectra.
`unannotated_amount` requests an absolute count. The larger request is capped by
available unannotated spectra. `random_seed` makes selection reproducible;
`row_width` controls merged spatial layout.

### Verify merged annotations

```python
reader = SQLiteAnnotationReader(
    "data/tutorial_workspace/datasets/catalog.sqlite",
    merged_dataset_id="merged-id",
)
source = reader.get_spectrum_metadata(0)
molecules = reader.get_spectrum_annotations(0)
```
