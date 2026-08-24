#!/usr/bin/env bash

# Submit bounded Slurm batches sequentially and finalize one staged campaign.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 [--restart] <campaign-id>" >&2
    exit 2
fi

restart=false
if [[ $# -eq 2 ]]; then
    if [[ $1 != "--restart" ]]; then
        echo "Unknown option: $1" >&2
        exit 2
    fi
    restart=true
    CAMPAIGN_ID=$2
else
    CAMPAIGN_ID=$1
fi

if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Campaign ID may contain only letters, numbers, underscores, and hyphens." >&2
    exit 2
fi

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
RUN_DIRECTORY=${HOME}/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}
TASK_COUNT_FILE=${RUN_DIRECTORY}/task-count
NEXT_TASK_FILE=${RUN_DIRECTORY}/next-task-index
TASK_JOB_HISTORY=${RUN_DIRECTORY}/task-array-job-ids
FINALIZER_JOB_FILE=${RUN_DIRECTORY}/finalizer-job-id
TASK_LIMIT=6
PARALLELISM=3
SBATCH_EXPORT="NIL,CAMPAIGN_ID=${CAMPAIGN_ID},REPOSITORY_ROOT=${REPOSITORY_ROOT},HOME=${HOME},USER=${USER},PATH=/usr/local/bin:/usr/bin:/bin"

if [[ ! -f "${TASK_COUNT_FILE}" ]]; then
    echo "Missing ${TASK_COUNT_FILE}. Complete staging before orchestration." >&2
    exit 1
fi
task_count=$(<"${TASK_COUNT_FILE}")
if [[ ! "${task_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid task count: ${task_count}" >&2
    exit 1
fi

# Restart only cancels arrays recorded for this campaign and preserves staged data.
if ${restart}; then
    if [[ -f "${TASK_JOB_HISTORY}" ]]; then
        while read -r job_id _; do
            if [[ "${job_id}" =~ ^[0-9]+$ ]] && squeue --noheader --jobs "${job_id}" | grep -q .; then
                scancel "${job_id}"
                echo "Cancelled recorded campaign array ${job_id}."
            fi
        done <"${TASK_JOB_HISTORY}"
    fi
    rm -f -- "${NEXT_TASK_FILE}" "${TASK_JOB_HISTORY}" "${FINALIZER_JOB_FILE}"
fi

next_task=0
if [[ -f "${NEXT_TASK_FILE}" ]]; then
    next_task=$(<"${NEXT_TASK_FILE}")
fi
if [[ ! "${next_task}" =~ ^[0-9]+$ ]] || (( next_task > task_count )); then
    echo "Invalid next task index: ${next_task}" >&2
    exit 1
fi

wait_for_job_completion() {
    local job_id=$1
    while squeue --noheader --jobs "${job_id}" | grep -q .; do
        sleep 30
    done
    # Allow Slurm accounting to release the completed array before the next submission.
    sleep 10
}

assert_completed_batch() {
    local first_task=$1
    local last_task=$2
    local task_index status_file
    for ((task_index = first_task; task_index <= last_task; task_index += 1)); do
        printf -v status_file '%s/plan/status/task_%06d.yaml' "${RUN_DIRECTORY}" "${task_index}"
        if [[ ! -f "${status_file}" ]] || ! grep -qx 'status: completed' "${status_file}"; then
            echo "Task ${task_index} did not complete successfully: ${status_file}" >&2
            exit 1
        fi
    done
}

while (( next_task < task_count )); do
    # The QoS permits six submitted tasks; wait rather than competing with other user jobs.
    while squeue --noheader --user "${USER}" | grep -q .; do
        sleep 30
    done

    last_task=$((next_task + TASK_LIMIT - 1))
    if (( last_task >= task_count )); then
        last_task=$((task_count - 1))
    fi

    job_id=$(sbatch --parsable \
        --array="${next_task}-${last_task}%${PARALLELISM}" \
        --export="${SBATCH_EXPORT}" \
        "${REPOSITORY_ROOT}/assets/scripts/entropy_predictive_task_array.sbatch")
    job_id=${job_id%%;*}
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Could not parse the task-array job ID: ${job_id}" >&2
        exit 1
    fi

    printf '%s %s-%s\n' "${job_id}" "${next_task}" "${last_task}" >>"${TASK_JOB_HISTORY}"
    echo "Submitted array ${job_id}: tasks ${next_task}-${last_task}."
    wait_for_job_completion "${job_id}"
    assert_completed_batch "${next_task}" "${last_task}"

    next_task=$((last_task + 1))
    printf '%s\n' "${next_task}" >"${NEXT_TASK_FILE}"
done

job_id=$(sbatch --parsable \
    --export="${SBATCH_EXPORT}" \
    "${REPOSITORY_ROOT}/assets/scripts/entropy_predictive_finalize.sbatch")
job_id=${job_id%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse the finalizer job ID: ${job_id}" >&2
    exit 1
fi
printf '%s\n' "${job_id}" >"${FINALIZER_JOB_FILE}"
echo "Submitted finalizer ${job_id}."
wait_for_job_completion "${job_id}"
echo "Campaign ${CAMPAIGN_ID} completed."
