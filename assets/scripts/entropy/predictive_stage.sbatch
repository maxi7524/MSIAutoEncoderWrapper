#!/usr/bin/env bash
#SBATCH --job-name=msi-stage
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --nodelist=asusgpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# Stage one workspace copy on the execution node and materialize a campaign plan.
set -euo pipefail

if (( $# == 1 )); then
    # Legacy invocation preserved for the established kidney predictive workflow.
    CAMPAIGN_ID=$1
    EXPERIMENT_CONFIG_INPUT=assets/experiments/08_26/23_08_26_architecture_predictive/architecture_predictive_experiment.yaml
    WORKSPACE_INPUT=data/kidney_workspace
    RUN_DIRECTORY_INPUT="${HOME}/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}"
elif (( $# == 3 || $# == 4 )); then
    CAMPAIGN_ID=$1
    EXPERIMENT_CONFIG_INPUT=$2
    WORKSPACE_INPUT=$3
    RUN_DIRECTORY_INPUT=${4:-"${HOME}/entropy-runs/${CAMPAIGN_ID}"}
else
    echo "Usage: sbatch $0 <campaign-id> [<experiment-yaml> <workspace-dir> [run-directory]]" >&2
    exit 2
fi

if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Campaign ID may contain only letters, numbers, underscores, and hyphens." >&2
    exit 2
fi

REPOSITORY_ROOT=${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "${REPOSITORY_ROOT}" || ! -f "${REPOSITORY_ROOT}/pyproject.toml" ]]; then
    echo "Set REPOSITORY_ROOT to the repository containing pyproject.toml." >&2
    exit 2
fi
REPOSITORY_ROOT=$(cd "${REPOSITORY_ROOT}" && pwd)
PYTHON=${REPOSITORY_ROOT}/.venv/bin/python

resolve_existing_path() {
    local input_path=$1
    local candidate_path
    if [[ "${input_path}" = /* ]]; then
        candidate_path=${input_path}
    else
        candidate_path=${REPOSITORY_ROOT}/${input_path}
    fi
    cd "$(dirname "${candidate_path}")" && printf '%s/%s\n' "$(pwd)" "$(basename "${candidate_path}")"
}

SOURCE_CONFIG=$(resolve_existing_path "${EXPERIMENT_CONFIG_INPUT}")
SOURCE_WORKSPACE=$(resolve_existing_path "${WORKSPACE_INPUT}")
if [[ "${RUN_DIRECTORY_INPUT}" = /* ]]; then
    RUN_DIRECTORY=${RUN_DIRECTORY_INPUT%/}
else
    RUN_DIRECTORY=${REPOSITORY_ROOT}/${RUN_DIRECTORY_INPUT%/}
fi

NODE_LOCAL_DIRECTORY=/tmp/${USER}/msi-wrapper/${CAMPAIGN_ID}
LOCAL_WORKSPACE=${NODE_LOCAL_DIRECTORY}/workspace
CAMPAIGN_FILE=${RUN_DIRECTORY}/entropy-campaign.env

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}. Run assets/scripts/entropy/setup_environment.sh first." >&2
    exit 1
fi
if [[ ! -f "${SOURCE_CONFIG}" ]]; then
    echo "Experiment YAML is missing: ${SOURCE_CONFIG}" >&2
    exit 1
fi
if [[ ! -d "${SOURCE_WORKSPACE}" ]]; then
    echo "Workspace directory is missing: ${SOURCE_WORKSPACE}" >&2
    exit 1
fi
if [[ -e "${NODE_LOCAL_DIRECTORY}" || -e "${RUN_DIRECTORY}" ]]; then
    echo "Campaign already exists. Choose a new campaign ID or run directory." >&2
    exit 1
fi

TASK_WALLTIME=$("${PYTHON}" - "${SOURCE_CONFIG}" <<'PY'
from pathlib import Path
import re
import sys

import yaml

with Path(sys.argv[1]).open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
value = config.get("execution", {}).get("entropy", {}).get("task_walltime", "01:00:00")
if not isinstance(value, str) or re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", value) is None:
    raise SystemExit("execution.entropy.task_walltime must use HH:MM:SS.")
print(value)
PY
)

"${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA on the staging node.")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
PY

# Single-copy node-local staging
## Use local NVMe rather than tmpfs so the workspace does not consume the job RAM cgroup.
source_bytes=$(du -sb "${SOURCE_WORKSPACE}" | awk '{print $1}')
available_bytes=$(df --block-size=1 --output=avail /tmp | awk 'NR == 2 {print $1}')
required_bytes=$((source_bytes + source_bytes / 10))
if (( available_bytes < required_bytes )); then
    echo "Insufficient /tmp capacity: need ${required_bytes} bytes, have ${available_bytes}." >&2
    exit 1
fi

mkdir -p "${NODE_LOCAL_DIRECTORY}" "${RUN_DIRECTORY}/logs"
trap 'rm -rf -- "${NODE_LOCAL_DIRECTORY}"' ERR
rsync -a --delete "${SOURCE_WORKSPACE}/" "${LOCAL_WORKSPACE}/"

# Persist campaign settings for the coordinator, task array, and finalizer.
{
    printf 'CAMPAIGN_ID=%q\n' "${CAMPAIGN_ID}"
    printf 'REPOSITORY_ROOT=%q\n' "${REPOSITORY_ROOT}"
    printf 'RUN_DIRECTORY=%q\n' "${RUN_DIRECTORY}"
    printf 'SOURCE_CONFIG=%q\n' "${SOURCE_CONFIG}"
    printf 'SOURCE_WORKSPACE=%q\n' "${SOURCE_WORKSPACE}"
    printf 'NODE_LOCAL_DIRECTORY=%q\n' "${NODE_LOCAL_DIRECTORY}"
    printf 'LOCAL_WORKSPACE=%q\n' "${LOCAL_WORKSPACE}"
    printf 'TASK_WALLTIME=%q\n' "${TASK_WALLTIME}"
} >"${CAMPAIGN_FILE}"

# Materialize descriptors against the same node-local workspace used by every task.
RUNTIME_CONFIG=${RUN_DIRECTORY}/runtime.yaml
"${PYTHON}" - "${SOURCE_CONFIG}" "${RUNTIME_CONFIG}" "${LOCAL_WORKSPACE}" <<'PY'
from pathlib import Path
import sys

import yaml

source_path, destination_path, workspace_path = map(Path, sys.argv[1:])
with source_path.open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)

factory_parameters = config["task"]["parameters"]["factory_parameters"]
factory_parameters["project_path"] = str(workspace_path)
factory_parameters["project_path_anchor"] = "working_directory"

with destination_path.open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY

"${PYTHON}" -m msi_autoencoder_wrapper.runtime.cli plan \
    "${RUNTIME_CONFIG}" \
    --output "${RUN_DIRECTORY}/plan"

# Record the task count for the login-node coordinator.
## The coordinator keeps each submitted array within the Entropy QoS limits.
task_count=$(find "${RUN_DIRECTORY}/plan/tasks" -maxdepth 1 -name 'task_*.yaml' -type f | wc -l)
if (( task_count < 1 )); then
    echo "Planning produced no task descriptors." >&2
    exit 1
fi
printf '%s\n' "${task_count}" >"${RUN_DIRECTORY}/task-count"
echo "Staging completed for ${CAMPAIGN_FILE}; ${task_count} tasks are ready."
echo "Run predictive_orchestrate.sh ${RUN_DIRECTORY} from the login node."
