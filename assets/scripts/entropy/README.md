# Entropy campaign workflow

These scripts run any materialized YAML experiment on Entropy without copying a
workspace once per task. They use the same relative workspace layout locally
and remotely. A campaign stores its durable execution artifacts under:

```text
data/<workspace>/
├── configs/entropy-runs/<campaign-id>/  # plan, task statuses, logs, runtime YAML
└── models/<context>/<campaign-id>__task_<index>/
```

The default remote host is `entropy`. It is an SSH alias defined by the user in
`~/.ssh/config`, not a hard-coded hostname or account name.

## User-facing scripts

Run scripts `01` through `06` on the Entropy login node. Run script `07` on the
local computer after a completed campaign.

| Script | Role |
| --- | --- |
| `01_setup_environment.sh` | Create or update the repository virtual environment. |
| `02_stage_campaign.sbatch` | Copy one workspace to node-local NVMe and materialize task descriptors. |
| `03_orchestrate_campaign.sh` | Submit bounded task arrays, verify status files, then finalize. |
| `04_run_campaign_sequence.sh` | Stage and execute several campaigns sequentially, reusing node-local capacity. |
| `05_task_array.sbatch` | Internal worker: execute one task-array element. |
| `06_finalize_campaign.sbatch` | Internal worker: copy only campaign-scoped models back to the workspace. |
| `07_download_campaign.sh` | Download one completed campaign's models and execution artifacts. |

## Lifecycle

```text
workspace on /home
      │
      ├── 02: one rsync copy
      ▼
/tmp/$USER/msi-wrapper/<campaign>/workspace
      │
      ├── 03: arrays of at most 6 submitted tasks, at most 3 concurrent
      ▼
task logs + plan/status in workspace/configs/entropy-runs/<campaign>
      │
      ├── 06: copy only <campaign>__task_* model directories
      ▼
workspace/models/<context>/<campaign>__task_<index>
```

`/tmp` is node-local NVMe storage, not RAM and not GPU memory. Every task in a
campaign reads the same staged workspace. The task arrays do not make another
workspace copy.

## Single campaign

### Staging campaign 
Staging creates all experiments, by creating common instances and preparing run folders. 

On the Entropy login node, from the repository root:

```bash
cd ~/repositories/MSIAutoEncoderWrapper
git pull --ff-only

SCRIPTS=assets/scripts/entropy
bash "${SCRIPTS}/01_setup_environment.sh"

# REMARK: Here put your workspace path 
WORKSPACE=data/kidney_workspace
# REMARK: Here put your `yaml` config path 
EXPERIMENT_YAML=assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/bce_baseline_experiment.yaml
# REMARK: Here put your experiment name 
CAMPAIGN_ID=bce-baseline-$(date +%Y%m%d)-01
RUN_DIRECTORY="${WORKSPACE}/configs/entropy-runs/${CAMPAIGN_ID}"

sbatch "${SCRIPTS}/02_stage_campaign.sbatch" \
  "${CAMPAIGN_ID}" \
  "${EXPERIMENT_YAML}" \
  "${WORKSPACE}"
```

### Runner coordinator 
Wait for staging to finish before starting the coordinator. 

```bash
ls "${RUN_DIRECTORY}/task-count"
cat "${RUN_DIRECTORY}/task-count"

nohup bash "${SCRIPTS}/03_orchestrate_campaign.sh" "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator.log" 2>&1 &
```

The three required inputs are:

1. `CAMPAIGN_ID` — a new, unique label for this execution.
2. `EXPERIMENT_YAML` — the configuration defining the dataset, model and task grid.
3. `WORKSPACE` — one relative workspace path, for example `data/kidney_workspace`.

The run directory is derived automatically from `WORKSPACE`. Do not reuse a
campaign ID: staging rejects an existing run directory and model destinations.

> Remark:
> If some runs are invalid, there will be not executed. 

## Several campaigns overnight

The sequence launcher runs one campaign at a time. This matters because a
workspace copy can occupy roughly 29 GB of the selected node's local `/tmp`.

```bash
WORKSPACE=data/kidney_workspace
RUN_ROOT="${WORKSPACE}/configs/entropy-runs"
mkdir -p "${RUN_ROOT}"

nohup bash "${SCRIPTS}/04_run_campaign_sequence.sh" \
  "${WORKSPACE}" \
  bce-baseline-YYYYMMDD-01 assets/experiments/<experiment>/bce_baseline_experiment.yaml \
  contractive-YYYYMMDD-01 assets/experiments/<experiment>/contractive_metric_weight_experiment.yaml \
  > "${RUN_ROOT}/sequence-YYYYMMDD.log" 2>&1 &
```

## Monitoring and validation

### Monitoring 

```bash
# Check current setup 
squeue -u "$USER"
tail -n 50 "${RUN_DIRECTORY}/orchestrator.log"
## REMARK: Here put ids of task and jobs 
tail -n 50 "${RUN_DIRECTORY}/logs/task_<array-job-id>_<task-index>.log"
### Example
### For `Submitted array 12508: tasks 0-4.` we can put 
tail -n 50 "${RUN_DIRECTORY}/logs/task_12508_2.log"

# Check completed status
completed=$(grep -lE '^[[:space:]]*status: completed$' \
  "${RUN_DIRECTORY}"/plan/status/task_*.yaml | wc -l)
failed=$(grep -lE '^[[:space:]]*status: failed$' \
  "${RUN_DIRECTORY}"/plan/status/task_*.yaml | wc -l)
printf 'completed=%s failed=%s\n' "$completed" "$failed"
```

For a completed campaign, `completed` equals `task-count`, `failed=0`, and the
finalizer job ID recorded in `${RUN_DIRECTORY}/finalizer-job-id` has Slurm state
`COMPLETED` with exit code `0:0`.

### Problem handling: Resume after coordinator stopped 

To resume after the coordinator process stops, do not create a new campaign:

```bash
nohup bash "${SCRIPTS}/03_orchestrate_campaign.sh" --restart "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator-restart.log" 2>&1 &
```

## Download to the local repository

On the local computer, from the same repository checkout:

```bash
WORKSPACE=data/kidney_workspace
CAMPAIGN_ID=bce-baseline-YYYYMMDD-01

bash assets/scripts/entropy/07_download_campaign.sh \
  "${WORKSPACE}" \
  "${CAMPAIGN_ID}"
```

Optional arguments are the SSH alias and the remote repository path relative to
the remote home directory:

```bash
bash assets/scripts/entropy/07_download_campaign.sh \
  data/kidney_workspace \
  bce-baseline-YYYYMMDD-01 \
  entropy \
  repositories/MSIAutoEncoderWrapper
```

The script downloads only `models/**/<campaign-id>__task_*` and the matching
`configs/entropy-runs/<campaign-id>` directory. It does not copy datasets or
models from previous campaigns.

## Values that may need changing

| What changes | Where |
| --- | --- |
| Dataset, split, model, losses, repetitions | Experiment YAML. |
| Workspace and campaign label | The three input variables in the launch command. |
| Remote SSH alias or remote repository location | Optional arguments to `07_download_campaign.sh`. |
| Partition, QoS, node, GPU, CPUs, staging walltime | `02_stage_campaign.sbatch`. |
| Task GPU/CPU allocation and default task walltime | `05_task_array.sbatch`; per-experiment walltime can be set as `execution.entropy.task_walltime` in the YAML. |
| Finalizer allocation | `06_finalize_campaign.sbatch`. |
| Account concurrency policy | `TASK_LIMIT` and `PARALLELISM` in `03_orchestrate_campaign.sh`. |

For the currently observed QoS, keep `TASK_LIMIT=6` and `PARALLELISM=3` unless
`sacctmgr show qos <qos-name> format=Name,MaxSubmitJobsPU,MaxJobsPU,MaxTRESPerUser,MaxWall`
shows a different limit.
