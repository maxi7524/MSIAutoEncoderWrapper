#!/usr/bin/env bash
#SBATCH --job-name=msi-predictive-stage
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --nodelist=asusgpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# Stage one immutable dataset copy on asusgpu6 and materialize its task array.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 <campaign-id>" >&2
    exit 2
fi

CAMPAIGN_ID=$1
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
SOURCE_WORKSPACE=${REPOSITORY_ROOT}/data/kidney_workspace
SOURCE_CONFIG=${REPOSITORY_ROOT}/assets/experiments/08_26/23_08_26_architecture_predictive/architecture_predictive_experiment.yaml
RUN_DIRECTORY=${HOME}/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}
NODE_LOCAL_DIRECTORY=/tmp/${USER}/msi-wrapper/${CAMPAIGN_ID}
LOCAL_WORKSPACE=${NODE_LOCAL_DIRECTORY}/kidney_workspace

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}. Run assets/scripts/entropy_setup_environment.sh first." >&2
    exit 1
fi
"${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot access CUDA on asusgpu6.")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
PY
if [[ ! -f "${SOURCE_WORKSPACE}/datasets/kidney/kidney.imzML" ]]; then
    echo "Kidney dataset is missing from ${SOURCE_WORKSPACE}." >&2
    exit 1
fi
if [[ -e "${NODE_LOCAL_DIRECTORY}" || -e "${RUN_DIRECTORY}" ]]; then
    echo "Campaign already exists: ${CAMPAIGN_ID}. Choose a new campaign ID." >&2
    exit 1
fi

# Single-copy node-local staging
## Use local NVMe rather than tmpfs so the dataset does not consume the job's RAM cgroup limit.
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

# Materialize task descriptors against the shared node-local workspace.
RUNTIME_CONFIG=${RUN_DIRECTORY}/architecture_predictive_runtime.yaml
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

# Record the task count for a later login-node submission.
## Entropy permits only one submitted job per user, so this job must finish
## before the task array can be submitted.
task_count=$(find "${RUN_DIRECTORY}/plan/tasks" -maxdepth 1 -name 'task_*.yaml' -type f | wc -l)
if (( task_count < 1 )); then
    echo "Planning produced no task descriptors." >&2
    exit 1
fi
printf '%s\n' "${task_count}" >"${RUN_DIRECTORY}/task-count"
echo "Staging completed for campaign ${CAMPAIGN_ID}; ${task_count} tasks are ready."
echo "Run entropy_predictive_submit_task_array.sh ${CAMPAIGN_ID} from the login node."
