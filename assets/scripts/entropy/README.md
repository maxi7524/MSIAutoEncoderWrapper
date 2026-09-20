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

Run scripts `01` through `04`, including the `03_1` and `03_2` workers, on the Entropy login node. Run script `07` on the
local computer after a completed campaign.

| Script | Role |
| --- | --- |
| `01_setup_environment.sh` | Create or update the repository virtual environment. |
| `02_stage_campaign.sbatch` | Copy one workspace to node-local NVMe and materialize task descriptors. |
| `03_orchestrate_campaign.sh` | Submit bounded task arrays, verify status files, then finalize. |
| `04_run_campaign_sequence.sh` | Stage and execute several campaigns sequentially, reusing node-local capacity. |
| `03_1_task_array.sbatch` | Internal worker of `03`: execute one task-array element. |
| `03_2_finalize_campaign.sbatch` | Internal worker of `03`: copy only campaign-scoped models back to the workspace. |
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
SELECTED_NODE=asusgpu1  # The same node that was used for staging.

# Restart the persistent coordinator
## Completed task manifests are retained; only unfinished task indices are submitted.
nohup bash assets/scripts/entropy/03_orchestrate_campaign.sh \
  --restart "${RUN_DIRECTORY}" \
  --nodelist "${SELECTED_NODE}" \
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

# Choose exactly one of the following staging commands.

# Default-node staging
## Uses the node specified by `#SBATCH --nodelist` in 02_stage_campaign.sbatch.
## In the current script this is `asusgpu6`; use this only when that node has
## enough local `/tmp` capacity.
STAGING_ROOT="/tmp/${USER}/msi-wrapper"
export STAGING_ROOT REPOSITORY_ROOT="${PWD}"
sbatch \
  "${SCRIPTS}/02_stage_campaign.sbatch" \
  "${CAMPAIGN_ID}" \
  "${EXPERIMENT_YAML}" \
  "${WORKSPACE}"

# Node-specific staging
## Use this version after the node-capacity diagnostic below. It overrides the
## script default and submits the campaign to the selected node.

## This creates the node-local copy and plan; it does not begin model training.
SELECTED_NODE=asusgpu1
STAGING_ROOT="/tmp/${USER}/msi-wrapper"
export STAGING_ROOT REPOSITORY_ROOT="${PWD}"
sbatch --nodelist="${SELECTED_NODE}" \
  "${SCRIPTS}/02_stage_campaign.sbatch" \
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
## Use the same node that was selected for staging.
SELECTED_NODE=asusgpu1
nohup bash "${SCRIPTS}/03_orchestrate_campaign.sh" "${RUN_DIRECTORY}" \
  --nodelist "${SELECTED_NODE}" \
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

The optional first argument is `--nodelist NODE`; when provided, the same node
is used for staging, task arrays, and finalization. The next argument is
`WORKSPACE`. Every following two arguments form one campaign: first its unique
identifier, then its YAML path.

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
  --nodelist asusgpu1 \
  "${WORKSPACE}" \
  bce-baseline-YYYYMMDD-01 assets/experiments/<experiment>/bce_baseline_experiment.yaml \
  contractive-YYYYMMDD-01 assets/experiments/<experiment>/contractive_metric_weight_experiment.yaml \
  > "${RUN_ROOT}/sequence-YYYYMMDD.log" 2>&1 &
```

## Monitoring and validation

### How to find proper node
Staging needs one GPU and local `/tmp` space for the workspace copy. The
staging script requires the workspace size plus 10% free space:

```bash
PARTITION=common
# Here you need specify your QOS 
QOS=ms488923_common
WORKSPACE=data/kidney_workspace
STAGING_USER="$USER"
STAGING_ROOT="/tmp/${STAGING_USER}/msi-wrapper"

# Use the same capacity calculation as 02_stage_campaign.sbatch.
SOURCE_BYTES=$(du -sb "${WORKSPACE}" | awk '{print $1}')
REQUIRED_BYTES=$((SOURCE_BYTES + SOURCE_BYTES / 10))
printf 'workspace=%s bytes, required staging space=%s bytes\n' \
  "${SOURCE_BYTES}" "${REQUIRED_BYTES}"
```

#### Check GPU allocations known to Slurm

```bash
# Node state, total GPUs, memory and currently free memory.
sinfo -N -p "${PARTITION}" \
  -o '%N %T %G %m %e %c %f' \
  | sort -k2,2 -k1,1

# Exact configured and allocated resources per node.
for node in $(sinfo -h -N -p "${PARTITION}" -o '%N' | sort -u); do
  echo "=== ${node} ==="
  scontrol show node "${node}" \
    | tr ' ' '\n' \
    | grep -E 'NodeName=|State=|CfgTRES=|AllocTRES=|RealMemory=|TmpDisk='
done

# Jobs currently using GPU resources.
squeue -p "${PARTITION}" \
  -o '%.18i %.9T %.24j %.12u %.12b %.20R'
```

`State=IDLE` and an empty `AllocTRES` mean that Slurm currently sees no
allocation on a node. `TmpDisk=0` does not describe the mounted `/tmp`; it
only means that Slurm does not track that local filesystem.

#### Probe local `/tmp` on candidate nodes

The following probe temporarily allocates one GPU per idle node for up to two
minutes. `--export=NIL` avoids Slurm's user-environment retrieval problem;
only the required variables are passed explicitly. Because `NIL` also removes
the inherited `PATH`, the probe passes a minimal `PATH` explicitly. It also
uses `--ntasks=1` to produce exactly one record per node.

```bash
PROBE_DIR=$(mktemp -d /tmp/msi-node-probe-XXXXXX)
trap 'rm -rf -- "${PROBE_DIR}"' EXIT

for node in $(sinfo -h -N -p "${PARTITION}" -o '%N %T' \
  | awk '$2 == "idle" {print $1}'); do
  (
    timeout 30s srun \
      --partition="${PARTITION}" \
      --qos="${QOS}" \
      --nodelist="${node}" \
      --ntasks=1 \
      --gres=gpu:1 \
      --cpus-per-task=1 \
      --mem=1G \
      --time=00:02:00 \
      --export=NIL,PATH=/usr/local/bin:/usr/bin:/bin,STAGING_USER="${STAGING_USER}",REQUIRED_BYTES="${REQUIRED_BYTES}" \
      --quiet \
      /bin/bash -c '
        available=$(df --block-size=1 --output=avail /tmp | awk "NR == 2 {print \$1}")
        used_stage=$(du -sb "/tmp/${STAGING_USER}/msi-wrapper" 2>/dev/null | awk "{print \$1}")
        used_stage=${used_stage:-0}
        margin=$((available - REQUIRED_BYTES))
        gpu=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader | tr "\n" ";")
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
          "$(hostname)" "$available" "$REQUIRED_BYTES" "$margin" "$used_stage" "$gpu"
      ' >"${PROBE_DIR}/${node}.out" 2>"${PROBE_DIR}/${node}.err"
  ) &
done
wait

printf 'node\tavailable_bytes\trequired_bytes\tmargin_bytes\texisting_stage_bytes\tgpu\n'
for result in "${PROBE_DIR}"/*.out; do
  [ -s "${result}" ] && cat "${result}"
done | sort -t $'\t' -k4,4nr | column -t -s $'\t'

for error in "${PROBE_DIR}"/*.err; do
  if [ -s "${error}" ]; then
    printf '\nProbe errors from %s:\n' "$(basename "${error%.err}")" >&2
    cat "${error}" >&2
  fi
done
```

Choose a node with a positive `margin`, preferably the largest one. A node
with a negative margin will fail before `rsync` starts. Existing staging data
is already included in `df` and must not be added to `REQUIRED_BYTES` again.

#### Submit staging on the selected node

Pass the selected node explicitly. For this Entropy cluster, export the
staging variables in the current shell and do not pass an `--export` option to
the staging `sbatch` command. Both `--export=ALL,...` and
`--export=NIL,...` caused direct staging jobs to enter
`user_env_retrieval_failed_requeued_held`, while the inherited environment
worked. This is cluster-specific behaviour; the local-node probe above keeps
its explicit `--export=NIL,...` because it also passes a minimal `PATH` to
`srun` and was verified independently.

```bash
SELECTED_NODE=asusgpu1
STAGING_ROOT="/tmp/${USER}/msi-wrapper"
export STAGING_ROOT REPOSITORY_ROOT="${PWD}"

STAGE_JOB_ID=$(sbatch --parsable \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --nodelist="${SELECTED_NODE}" \
  "${SCRIPTS}/02_stage_campaign.sbatch" \
  "${CAMPAIGN_ID}" \
  "${EXPERIMENT_YAML}" \
  "${WORKSPACE}")

echo "Submitted staging job: ${STAGE_JOB_ID} on ${SELECTED_NODE}"
squeue -j "${STAGE_JOB_ID}"
```

If a staging job is held with `user_env_retrieval_failed_requeued_held`,
cancel it and resubmit without `--export`:

```bash
scancel "${STAGE_JOB_ID}" 2>/dev/null || true
```

This error is emitted before the staging script can create its normal log.
It does not indicate insufficient `/tmp` capacity. Check the actual staging
result with `sacct` and verify that `${RUN_DIRECTORY}/task-count` exists before
starting the orchestrator.

After completion, verify the staging result before starting the coordinator:

```bash
sacct -j "${STAGE_JOB_ID}" \
  --format=JobID,JobName%25,State,ExitCode,Elapsed,NodeList,Reason
cat "slurm-${STAGE_JOB_ID}.out"
ls "${RUN_DIRECTORY}/task-count"
```

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
  --nodelist "${SELECTED_NODE}" \
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
| Task GPU/CPU allocation and default task walltime | `03_1_task_array.sbatch`; per-experiment walltime can be set as `execution.entropy.task_walltime` in the YAML. |
| Finalizer allocation | `03_2_finalize_campaign.sbatch`. |
| Account concurrency policy | `TASK_LIMIT` and `PARALLELISM` in `03_orchestrate_campaign.sh`. |

For the currently observed QoS, keep `TASK_LIMIT=6` and `PARALLELISM=3` unless
`sacctmgr show qos <qos-name> format=Name,MaxSubmitJobsPU,MaxJobsPU,MaxTRESPerUser,MaxWall`
shows a different limit.
