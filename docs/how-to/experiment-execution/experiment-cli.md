# Run the experiment CLI

The `msi-wrapper` command validates, materializes, executes, and reports an
experiment campaign.

## Purpose and available operations

### Public commands

The public commands are `validate`, `plan`, `run`, and `report`. Internal `task`
and `finalize` commands are generated for local workers and Slurm jobs.

### Materialized output

Planning writes resolved task YAML files and `resolved-experiment.yaml` before
external work begins.

## Detailed instructions

### Validate configuration and preflight

```bash
.venv/bin/msi-wrapper validate path/to/experiment.yaml
```

Validation checks schema fields and calls the configured preflight entrypoint.
No tasks are executed.

### Materialize a plan

```bash
.venv/bin/msi-wrapper plan path/to/experiment.yaml \
  --output workspace/executions/example-campaign
```

`--output` overrides `execution.work_directory`. Each task stores its grid
parameters, repetition, seed, resolved parameters, and entrypoint.

### Run locally or submit Slurm

```bash
.venv/bin/msi-wrapper run path/to/experiment.yaml
.venv/bin/msi-wrapper run path/to/experiment.yaml --dry-run
```

`--dry-run` materializes the plan without executing it. Local mode runs up to
`max_parallel_runs`. Slurm mode stages the plan, submits an array, and submits a
dependent finalizer that restores verified results and renders reports.

### Render reports separately

```bash
.venv/bin/msi-wrapper report path/to/experiment.yaml
.venv/bin/msi-wrapper report path/to/experiment.yaml --only report-name
```

The command writes a report manifest. It exits unsuccessfully when required
report execution fails.
