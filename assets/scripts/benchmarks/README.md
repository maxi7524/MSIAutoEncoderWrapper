# Benchmarks

This directory contains standalone, reproducible performance probes for MSI
data loading, preprocessing, and complete experiment campaigns.

## Campaign benchmark

`benchmark_campaigns.py` compares any two runtime campaign YAML files which
provide a `plan_entrypoint` and a `test_entrypoint`. It does not assume an
autoencoder or a particular criterion. For campaigns containing
`ContractiveLoss`, it additionally reports the Jacobian operation formula:
exact Frobenius VJPs, Hutchinson VJPs, or spectral JVP/VJP power iterations.

The output Markdown contains, for both YAMLs:

- configuration loading and grid expansion;
- runtime component resolution;
- a real bounded training probe from the configuration's own test entrypoint;
- wall-clock mean, standard deviation, observed P95, and two-sided 95%
  Student-t confidence intervals;
- process RSS delta, CUDA peak allocation, symbolic complexity, and an
  alternative/reference comparison.

Run it on the same node type and GPU partition intended for training:

```bash
# Repository and Python environment
## Run setup after a fresh clone or after Python dependencies change.
cd ~/repositories/MSIAutoEncoderWrapper
git pull --ff-only

# Benchmark inputs
## REMARK: Here put the baseline YAML configuration path.
REFERENCE_YAML=assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/contractive_metric_weight_experiment.yaml
## REMARK: Here put the candidate YAML configuration path.
ALTERNATIVE_YAML=assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/fisher_rao_spectral_hinge_experiment.yaml
## REMARK: Here put the Markdown destination for the numerical comparison table.
OUTPUT_MARKDOWN=assets/experiments/autoencoder_architecture/report/campaign-benchmark.md
## REMARK: Here put independent measurements per stage; use at least 2 for a 95% CI.
BENCHMARK_REPEATS=3
## REMARK: Here put the planned training device on the allocated node.
DEVICE=cuda
## REMARK: Here put the target number of train spectra for runtime extrapolation.
PROJECTED_SPECTRA=500000

# Run campaign benchmark
## The command runs bounded probes; it does not persist trained models.
UV_CACHE_DIR="$HOME/.cache/uv" uv run --extra cu118 python \
  assets/scripts/benchmarks/benchmark_campaigns.py \
  "${REFERENCE_YAML}" \
  "${ALTERNATIVE_YAML}" \
  "${OUTPUT_MARKDOWN}" \
  --repeats "${BENCHMARK_REPEATS}" \
  --device "${DEVICE}" \
  --projected-spectra "${PROJECTED_SPECTRA}"

# Shorter example
uv run --extra cu118 python \
  assets/scripts/benchmarks/benchmark_campaigns.py \
  assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/contractive_metric_weight_experiment.yaml \
  assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/fisher_rao_spectral_hinge_experiment.yaml \
  assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/campaign-benchmark.md \
  --repeats 3 --device cuda --projected-spectra 500000
```

`--repeats` is the number of independent benchmark observations per stage; it
does not change YAML repetitions or train models. It must be at least `2`
because every table row includes a confidence interval. `--projected-spectra`
linearly scales the probe-based estimate to a requested number of train spectra
and reports the current split's train spectra and planned batch count.
Alternatively, `--projected-input-gib` scales by raw MSI file size. Use at most
one projection option.

The train probe uses the campaign's `test_entrypoint`, which is expected to
bound itself to a small number of batches. It does not write a trained model.
Use `--device cuda` on Slurm worker nodes to obtain peak allocation for the
actual allocated GPU.
