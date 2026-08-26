#!/usr/bin/env bash
#SBATCH --job-name=msi-task
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --nodelist=asusgpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

# Execute one materialized task against the campaign workspace snapshot.
set -euo pipefail

if [[ -z "${CAMPAIGN_FILE:-}" || ! -f "${CAMPAIGN_FILE}" ]]; then
    echo "CAMPAIGN_FILE must name the campaign settings written by staging." >&2
    exit 2
fi
source "${CAMPAIGN_FILE}"

if [[ ! "${CAMPAIGN_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Campaign settings contain an invalid campaign ID." >&2
    exit 2
fi
if [[ ! -f "${REPOSITORY_ROOT:-}/pyproject.toml" ]]; then
    echo "Campaign settings contain an invalid repository root." >&2
    exit 2
fi

PYTHON=${REPOSITORY_ROOT}/.venv/bin/python
TASK_FILE=$(printf '%s/plan/tasks/task_%06d.yaml' "${RUN_DIRECTORY}" "${SLURM_ARRAY_TASK_ID}")
LOG_FILE=${RUN_DIRECTORY}/logs/task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log

if [[ ! -x "${PYTHON}" || ! -f "${TASK_FILE}" ]]; then
    echo "Missing environment or task file for campaign ${CAMPAIGN_ID}." >&2
    exit 1
fi

exec >"${LOG_FILE}" 2>&1
"${PYTHON}" -m msi_autoencoder_wrapper.runtime.cli task "${TASK_FILE}"
