#!/usr/bin/env bash

# Submit same-node result synchronization after the predictive array has completed.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <campaign-id>" >&2
    exit 2
fi

CAMPAIGN_ID=$1
if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Campaign ID may contain only letters, numbers, underscores, and hyphens." >&2
    exit 2
fi

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIRECTORY=${HOME}/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}
TASK_JOB_FILE=${RUN_DIRECTORY}/task-array-job-id

if [[ ! -f "${TASK_JOB_FILE}" ]]; then
    echo "Missing ${TASK_JOB_FILE}. Submit the task array first." >&2
    exit 1
fi
if squeue --noheader --user "${USER}" | grep -q .; then
    echo "Wait until all of your current Slurm jobs finish before submitting the finalizer." >&2
    exit 1
fi

job_id=$(sbatch --parsable \
    --export="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},REPOSITORY_ROOT=${REPOSITORY_ROOT}" \
    "${REPOSITORY_ROOT}/assets/scripts/entropy_predictive_finalize.sbatch")
job_id=${job_id%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse the finalizer job ID: ${job_id}" >&2
    exit 1
fi

printf '%s\n' "${job_id}" >"${RUN_DIRECTORY}/finalizer-job-id"
echo "Submitted finalizer ${job_id} for campaign ${CAMPAIGN_ID}."
