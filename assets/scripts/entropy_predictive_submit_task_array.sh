#!/usr/bin/env bash

# Submit the next bounded predictive task-array batch after staging completes.
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
TASK_JOB_HISTORY=${RUN_DIRECTORY}/task-array-job-ids
MAX_SUBMITTED_TASKS=6

if [[ ! -f "${TASK_COUNT_FILE}" ]]; then
    echo "Missing ${TASK_COUNT_FILE}. Finish the staging job first." >&2
    exit 1
fi
if squeue --noheader --user "${USER}" | grep -q .; then
    echo "Wait until all of your current Slurm jobs finish before submitting the array." >&2
    exit 1
fi

task_count=$(<"${TASK_COUNT_FILE}")
if [[ ! "${task_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid task count: ${task_count}" >&2
    exit 1
fi
next_task=0
if [[ -f "${NEXT_TASK_FILE}" ]]; then
    next_task=$(<"${NEXT_TASK_FILE}")
fi
if [[ ! "${next_task}" =~ ^[0-9]+$ ]]; then
    echo "Invalid next task index: ${next_task}" >&2
    exit 1
fi
if (( next_task >= task_count )); then
    echo "All ${task_count} tasks have already been submitted." >&2
    exit 0
fi

last_task=$((next_task + MAX_SUBMITTED_TASKS - 1))
if (( last_task >= task_count )); then
    last_task=$((task_count - 1))
fi

job_id=$(sbatch --parsable \
    --array="${next_task}-${last_task}%3" \
    --export="ALL,CAMPAIGN_ID=${CAMPAIGN_ID},REPOSITORY_ROOT=${REPOSITORY_ROOT}" \
    "${REPOSITORY_ROOT}/assets/scripts/entropy_predictive_task_array.sbatch")
job_id=${job_id%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse the task-array job ID: ${job_id}" >&2
    exit 1
fi

printf '%s\n' "$((last_task + 1))" >"${NEXT_TASK_FILE}"
printf '%s %s-%s\n' "${job_id}" "${next_task}" "${last_task}" >>"${TASK_JOB_HISTORY}"
echo "Submitted task array ${job_id}: tasks ${next_task}-${last_task} for campaign ${CAMPAIGN_ID}."
if (( last_task + 1 < task_count )); then
    echo "After this array finishes, run this script again for the next batch."
else
    echo "This was the last task batch; inspect results, then submit the finalizer."
fi
