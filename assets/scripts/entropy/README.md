# Entropy Slurm campaigns

This directory contains the Slurm workflow for YAML experiments using the
materialized-task runtime. A campaign stages one workspace copy on an Entropy
execution node, runs the generated tasks, and copies output artifacts back to
the source workspace.

## Inputs and campaign snapshot

`predictive_stage.sbatch` accepts a campaign identifier, an experiment YAML,
a workspace directory, and optionally a run directory. Relative paths are
resolved from the repository root. Staging writes the resolved paths to
`<run-directory>/entropy-campaign.env`; the coordinator, task array, and
finalizer all use this snapshot.

The YAML must contain
`task.parameters.factory_parameters.project_path`, because staging replaces it
with the node-local workspace path before materializing tasks.

Do not edit the source YAML or `entropy-campaign.env` after staging. Generate a
new campaign whenever model, dataset, loss, or training parameters change.

## Scripts

- `setup_environment.sh` creates or updates the repository `.venv`.
- `predictive_stage.sbatch` copies the workspace and materializes task YAMLs.
- `predictive_orchestrate.sh` submits bounded arrays, validates status files,
  and submits finalization.
- `predictive_task_array.sbatch` executes one materialized task.
- `predictive_finalize.sbatch` copies `models/` and `configs/` back to the
  source workspace and removes node-local staging.

## Commands

Run the commands on the Entropy login node from the repository root:

```bash
ssh entropy
cd ~/repositories/MSIAutoEncoderWrapper
git pull --ff-only

SCRIPTS=assets/scripts/entropy
bash "${SCRIPTS}/setup_environment.sh"
```

### Existing kidney predictive workflow

The previous one-argument commands remain supported:

```bash
CAMPAIGN_ID=predictive-YYYYMMDD-01

sbatch "${SCRIPTS}/predictive_stage.sbatch" "${CAMPAIGN_ID}"

RUN_DIRECTORY="$HOME/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}"
nohup bash "${SCRIPTS}/predictive_orchestrate.sh" "${CAMPAIGN_ID}" \
  > "${RUN_DIRECTORY}/orchestrator.log" 2>&1 &
```

This compatibility form selects:

```text
YAML:      assets/experiments/08_26/23_08_26_architecture_predictive/architecture_predictive_experiment.yaml
workspace: data/kidney_workspace
run root:  ~/entropy-runs/kidney-architecture-predictive/
```

### Another YAML or dataset workspace

Use the general staging form. Replace all four values with one coherent
experiment definition:

```bash
CAMPAIGN_ID=<unique-campaign-id>
EXPERIMENT_YAML=assets/experiments/<experiment>.yaml
WORKSPACE=data/<workspace>
RUN_DIRECTORY="$HOME/entropy-runs/${CAMPAIGN_ID}"

sbatch "${SCRIPTS}/predictive_stage.sbatch" \
  "${CAMPAIGN_ID}" \
  "${EXPERIMENT_YAML}" \
  "${WORKSPACE}" \
  "${RUN_DIRECTORY}"

nohup bash "${SCRIPTS}/predictive_orchestrate.sh" "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator.log" 2>&1 &
```

The source workspace must contain every relative data path used by the YAML.
For example, if the YAML refers to `datasets/liver/liver.imzML`, that file must
exist under `data/<workspace>/datasets/liver/`.

### Sequential night run for several YAMLs

One local workspace uses about 29 GB on `asusgpu6`, so campaigns must run one
after another. The sequence launcher stages one campaign, waits for its finalizer
to remove the node-local copy, and then starts the next one:

```bash
nohup bash "${SCRIPTS}/predictive_run_sequence.sh" \
  data/kidney_workspace \
  nnpu-controls-YYYYMMDD assets/experiments/08_26/23_08_26_architecture_predictive/nnpu_followup/nnpu_objective_controls.yaml \
  nnpu-priors-YYYYMMDD assets/experiments/08_26/23_08_26_architecture_predictive/nnpu_followup/nnpu_prior_sensitivity.yaml \
  nnpu-long-YYYYMMDD assets/experiments/08_26/23_08_26_architecture_predictive/nnpu_followup/nnpu_long_training.yaml \
  nnpu-masked30-YYYYMMDD assets/experiments/08_26/23_08_26_architecture_predictive/nnpu_followup/nnpu_masked30.yaml \
  > "$HOME/entropy-runs/nnpu-followup-YYYYMMDD-sequence.log" 2>&1 &
```

Use a date or another unique suffix once. Do not rerun this command with the
same campaign IDs, because staging intentionally rejects existing run directories.

## Execution

Staging copies the selected workspace once to
`/tmp/${USER}/msi-wrapper/<campaign-id>/workspace` on the configured node.
This is node-local NVMe storage, not RAM or GPU memory. All array elements read
the same workspace copy.

The coordinator submits at most six elements in a batch and limits the array to
three concurrent elements. It waits for a batch, verifies task status manifests,
then submits the next batch. After all tasks complete, finalization performs one
`rsync` of `models/` and `configs/` to the source workspace and removes the
campaign directory from `/tmp`.

The optional YAML setting `execution.entropy.task_walltime` controls the Slurm
walltime of task-array elements. It defaults to `01:00:00`; the long nnPU
campaign uses `02:30:00` because it performs 60 epochs plus train/test AP after
every epoch. Slurm charges elapsed time, not the unused remainder of this limit.

## Monitoring and restart

```bash
squeue -u "$USER"
tail -n 50 "${RUN_DIRECTORY}/orchestrator.log"
tail -n 50 "${RUN_DIRECTORY}/logs/task_<array-job-id>_<task-index>.log"
```

After an interrupted coordinator, use the same identifier or run directory:

```bash
nohup bash "${SCRIPTS}/predictive_orchestrate.sh" --restart "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator-restart.log" 2>&1 &
```

`--restart` cancels arrays recorded for that campaign, keeps completed task
status files, and resumes from the first incomplete task.

## Slurm settings

The three `.sbatch` files contain the partition, QoS, node, GPU, CPU, and wall
time directives. Update all three consistently when changing Entropy allocation
or hardware. `TASK_LIMIT=6` and `PARALLELISM=3` in
`predictive_orchestrate.sh` must not exceed the selected QoS limits.

```bash
sacctmgr show qos <qos-name> \
  format=Name,MaxSubmitJobsPU,MaxJobsPU,MaxTRESPerUser,MaxWall
```
