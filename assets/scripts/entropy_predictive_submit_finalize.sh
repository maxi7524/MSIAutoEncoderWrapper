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
TASK_COUNT_FILE=${RUN_DIRECTORY}/task-count
NEXT_TASK_FILE=${RUN_DIRECTORY}/next-task-index

if [[ ! -f "${TASK_COUNT_FILE}" || ! -f "${NEXT_TASK_FILE}" ]]; then
    echo "Submit every task batch before submitting the finalizer." >&2
    exit 1
fi
task_count=$(<"${TASK_COUNT_FILE}")
next_task=$(<"${NEXT_TASK_FILE}")
if [[ ! "${task_count}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${next_task}" =~ ^[0-9]+$ ]] || (( next_task < task_count )); then
    echo "Not every task has been submitted; finalization is blocked." >&2
    exit 1
fi
if squeue --noheader --user "${USER}" | grep -q .; then
    echo "Wait until all of your current Slurm jobs finish before submitting the finalizer." >&2
    exit 1
fi

job_id=$(sbatch --parsable \
    --export="NIL,CAMPAIGN_ID=${CAMPAIGN_ID},REPOSITORY_ROOT=${REPOSITORY_ROOT},HOME=${HOME},USER=${USER},PATH=/usr/local/bin:/usr/bin:/bin" \
    "${REPOSITORY_ROOT}/assets/scripts/entropy_predictive_finalize.sbatch")
job_id=${job_id%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse the finalizer job ID: ${job_id}" >&2
    exit 1
fi

printf '%s\n' "${job_id}" >"${RUN_DIRECTORY}/finalizer-job-id"
echo "Submitted finalizer ${job_id} for campaign ${CAMPAIGN_ID}."
