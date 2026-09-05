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

### Resume an existing campaign

Use this command only when staging already completed for the same campaign and
the coordinator stopped or the login session ended. It preserves completed task
statuses, cancels only arrays recorded for this campaign, and resumes from the
first task without `status: completed`. Do not run staging again.

```bash
# Existing campaign location
## REMARK: Use the same workspace and campaign ID that were used for staging.
WORKSPACE=data/kidney_workspace
CAMPAIGN_ID=<existing-campaign-id>
RUN_DIRECTORY="${WORKSPACE}/configs/entropy-runs/${CAMPAIGN_ID}"

# Restart the persistent coordinator
## Completed task manifests are retained; only unfinished task indices are submitted.
nohup bash assets/scripts/entropy/03_orchestrate_campaign.sh \
  --restart "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator-restart.log" 2>&1 &
```

### Define inputs nad stage new campaign 

Staging is the preparation step, it does not train models. It copies the
selected workspace once to node-local `/tmp`, changes the runtime YAML to use
that copy, expands the experiment grid into task descriptors, and writes the
durable campaign files under `configs/entropy-runs/<campaign-id>`.

On the Entropy login node, from the repository root:

```bash
# Repository and Python environment
## Run setup after a fresh clone or after Python dependencies change.
cd ~/repositories/MSIAutoEncoderWrapper
git pull --ff-only

SCRIPTS=assets/scripts/entropy
bash "${SCRIPTS}/01_setup_environment.sh"

# Campaign inputs
## REMARK: Here put your workspace path. It must be relative to both repositories.
WORKSPACE=data/kidney_workspace
## REMARK: Here put your YAML config path. It defines data, model, losses, and repetitions.
EXPERIMENT_YAML=assets/experiments/autoencoder_architecture/experiment_runs_configs/05_09_26_contractive_expaned/bce_baseline_experiment.yaml
## REMARK: Here put your experiment name. It must be new and scopes run files and model names.
CAMPAIGN_ID=bce-baseline-$(date +%Y%m%d)-01
## Derived automatically when the optional fourth staging argument is omitted.
RUN_DIRECTORY="${WORKSPACE}/configs/entropy-runs/${CAMPAIGN_ID}"

# Submit staging only
## This creates the node-local copy and plan; it does not begin model training.
sbatch "${SCRIPTS}/02_stage_campaign.sbatch" \
  "${CAMPAIGN_ID}" \
  "${EXPERIMENT_YAML}" \
  "${WORKSPACE}"
```

The three variable assignments are the only per-campaign launcher inputs:

1. `WORKSPACE` — the relative workspace path.
2. `EXPERIMENT_YAML` — the experiment definition.
3. `CAMPAIGN_ID` — a unique execution identifier.

Do not reuse a campaign ID. Staging rejects an existing run directory and the
finalizer rejects an existing model destination.

### Start the training coordinator 

Wait for staging to finish before starting the coordinator. `task-count` is
created only after the workspace copy and plan materialization succeeded.

```bash
# Verify that staging succeeded
## The expected task count is determined by the YAML grid and repetitions.
ls "${RUN_DIRECTORY}/task-count"
cat "${RUN_DIRECTORY}/task-count"

# Start training in the background
## The coordinator submits bounded arrays and writes its decisions to this log.
nohup bash "${SCRIPTS}/03_orchestrate_campaign.sh" "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator.log" 2>&1 &
```

This command performs the training. It submits the first batch of at most six
task-array elements with at most three running simultaneously. After the batch
leaves Slurm, it verifies every task status. Only then does it submit the next
batch. A failed task stops the coordinator and prevents finalization; use the
status file and task log to diagnose it before using `--restart`.

## Several campaigns orchestration

The sequence launcher runs one campaign at a time. This matters because a
workspace copy can occupy roughly 29 GB of the selected node's local `/tmp`.
For each `campaign-id YAML` pair, `04_run_campaign_sequence.sh` submits `02`,
waits for staging to succeed, runs `03` until its finalizer completes, and only
then stages the next pair. Therefore there is one staged workspace at a time;
within each campaign, `03` still runs up to three GPU tasks concurrently.

The first argument is `WORKSPACE`. Every following two arguments form one
campaign: first its unique identifier, then its YAML path.

```bash
# Shared workspace output root
## Every campaign below stores plan, statuses, and logs under this directory.
WORKSPACE=data/kidney_workspace
RUN_ROOT="${WORKSPACE}/configs/entropy-runs"
mkdir -p "${RUN_ROOT}"

# Sequential campaign list
## REMARK: After WORKSPACE, every two arguments are <campaign-id> <experiment-yaml>.
## The next campaign starts only after the preceding campaign finalizer succeeds.
nohup bash "${SCRIPTS}/04_run_campaign_sequence.sh" \
  "${WORKSPACE}" \
  bce-baseline-YYYYMMDD-01 assets/experiments/<experiment>/bce_baseline_experiment.yaml \
  contractive-YYYYMMDD-01 assets/experiments/<experiment>/contractive_metric_weight_experiment.yaml \
  > "${RUN_ROOT}/sequence-YYYYMMDD.log" 2>&1 &
```

## Monitoring and validation

### How to find CAMPAIGN_ID
All campaign ids cna be found in entropy runs folder

```bash
# Define workspace
WORKSPACE=data/kidney_workspace
RUN_ROOT="${WORKSPACE}/configs/entropy-runs"

find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort

# Find recent campaign
CAMPAIGN_ID=$(find "${RUN_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-)

echo "${CAMPAIGN_ID}"

RUN_DIRECTORY="${RUN_ROOT}/${CAMPAIGN_ID}"
``` 

### Monitor an active campaign 

```bash
# Scheduler state
## Shows queued and running jobs for this account.
squeue -u "$USER"
# Campaign-level log
## Shows submitted batches, validation failures, and finalizer submission.
tail -n 50 "${RUN_DIRECTORY}/orchestrator.log"
# One training task log
## REMARK: Replace both placeholders with an array job ID and a task index.
tail -n 50 "${RUN_DIRECTORY}/logs/task_<array-job-id>_<task-index>.log"
### Example: after `Submitted array 12508: tasks 0-4.` inspect task index 2.
tail -n 50 "${RUN_DIRECTORY}/logs/task_12508_2.log"

# Task-status summary
## This can be executed while training is active or after it ends.
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

The restart command is shown at the beginning of this section. It is repeated
here only as a reminder: resume an existing campaign; do not create a new one.

```bash
nohup bash "${SCRIPTS}/03_orchestrate_campaign.sh" --restart "${RUN_DIRECTORY}" \
  > "${RUN_DIRECTORY}/orchestrator-restart.log" 2>&1 &
```

## Download to the local repository

On the local computer, from the same repository checkout:

```bash

# Local campaign selection
## REMARK: Use the same workspace-relative path and completed campaign ID as on Entropy.
WORKSPACE=data/kidney_workspace
CAMPAIGN_ID=bce-baseline-YYYYMMDD-01

# Download models and campaign artifacts
## The default SSH host is the `entropy` alias from ~/.ssh/config.
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
