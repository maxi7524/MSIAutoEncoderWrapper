#!/usr/bin/env bash
#SBATCH --job-name=msi-predictive
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --nodelist=asusgpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# Execute one planned predictive task against the shared node-local dataset.
set -euo pipefail

if [[ -z "${CAMPAIGN_ID:-}" ]]; then
    echo "CAMPAIGN_ID must be exported by the staging job." >&2
    exit 2
fi

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${REPOSITORY_ROOT}/.venv/bin/python
RUN_DIRECTORY=${HOME}/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}
TASK_FILE=$(printf '%s/plan/tasks/task_%06d.yaml' "${RUN_DIRECTORY}" "${SLURM_ARRAY_TASK_ID}")
LOG_FILE=${RUN_DIRECTORY}/logs/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

if [[ ! -x "${PYTHON}" || ! -f "${TASK_FILE}" ]]; then
    echo "Missing environment or task file for campaign ${CAMPAIGN_ID}." >&2
    exit 1
fi

exec >"${LOG_FILE}" 2>&1
"${PYTHON}" -m msi_autoencoder_wrapper.runtime.cli task "${TASK_FILE}"
