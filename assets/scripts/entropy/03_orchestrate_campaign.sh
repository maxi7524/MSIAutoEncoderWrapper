#!/usr/bin/env bash

# Submit bounded Slurm batches sequentially and finalize one staged campaign.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 [--restart] <run-directory>" >&2
    exit 2
fi

restart=false
if [[ $# -eq 2 ]]; then
    if [[ $1 != "--restart" ]]; then
        echo "Unknown option: $1" >&2
        exit 2
    fi
    restart=true
    RUN_DIRECTORY_INPUT=$2
else
    RUN_DIRECTORY_INPUT=$1
fi

if [[ -d "${RUN_DIRECTORY_INPUT}" ]]; then
    REQUESTED_RUN_DIRECTORY=$(cd "${RUN_DIRECTORY_INPUT}" && pwd)
elif [[ "${RUN_DIRECTORY_INPUT}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    # Compatibility invocation for the default kidney workspace.
    SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    REPOSITORY_ROOT_DEFAULT=$(cd "${SCRIPT_DIRECTORY}/../../.." && pwd)
    REQUESTED_RUN_DIRECTORY=${REPOSITORY_ROOT_DEFAULT}/data/kidney_workspace/configs/entropy-runs/${RUN_DIRECTORY_INPUT}
else
    echo "Run directory is missing: ${RUN_DIRECTORY_INPUT}" >&2
    exit 1
fi
if [[ ! -d "${REQUESTED_RUN_DIRECTORY}" ]]; then
    echo "Run directory is missing: ${REQUESTED_RUN_DIRECTORY}" >&2
    exit 1
fi
CAMPAIGN_FILE=${REQUESTED_RUN_DIRECTORY}/entropy-campaign.env
if [[ ! -f "${CAMPAIGN_FILE}" ]]; then
    echo "Campaign settings are missing: ${CAMPAIGN_FILE}" >&2
    exit 1
fi

# The staging job writes this file. It binds every later job to one workspace snapshot.
source "${CAMPAIGN_FILE}"
if [[ ! "${CAMPAIGN_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Campaign settings contain an invalid campaign ID." >&2
    exit 1
fi
if [[ "${RUN_DIRECTORY:-}" != "${REQUESTED_RUN_DIRECTORY}" ]] || [[ ! -f "${REPOSITORY_ROOT:-}/pyproject.toml" ]]; then
    echo "Campaign settings contain an invalid repository or run directory." >&2
    exit 1
fi
RUN_DIRECTORY=${REQUESTED_RUN_DIRECTORY}

TASK_COUNT_FILE=${RUN_DIRECTORY}/task-count
NEXT_TASK_FILE=${RUN_DIRECTORY}/next-task-index
TASK_JOB_HISTORY=${RUN_DIRECTORY}/task-array-job-ids
FINALIZER_JOB_FILE=${RUN_DIRECTORY}/finalizer-job-id
# These values must not exceed the Entropy QoS submission and GPU limits.
TASK_LIMIT=6
PARALLELISM=3
TASK_WALLTIME=${TASK_WALLTIME:-01:00:00}

if [[ ! "${TASK_WALLTIME}" =~ ^[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]$ ]]; then
    echo "Campaign settings contain an invalid task walltime: ${TASK_WALLTIME}" >&2
    exit 1
fi

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
            if [[ "${job_id}" =~ ^[0-9]+$ ]] && squeue --noheader --jobs "${job_id}" 2>/dev/null | grep -q .; then
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
    while squeue --noheader --jobs "${job_id}" 2>/dev/null | grep -q .; do
        sleep 30
    done
    # Allow Slurm accounting to release the completed array before the next submission.
    sleep 10
}

# Finalizer verification
## Query Slurm accounting after a job leaves the scheduler queue.
assert_job_completed_successfully() {
    local job_id=$1
    local state=
    for _ in $(seq 1 12); do
        state=$(sacct --noheader --allocations --jobs "${job_id}" --format=State \
            | awk 'NR == 1 {print $1}')
        if [[ -n "${state}" ]]; then
            break
        fi
        sleep 5
    done
    if [[ "${state}" != COMPLETED ]]; then
        echo "Job ${job_id} ended with state '${state:-unknown}'." >&2
        exit 1
    fi
}

task_completed() {
    local task_index=$1
    local status_file
    printf -v status_file '%s/plan/status/task_%06d.yaml' "${RUN_DIRECTORY}" "${task_index}"
    [[ -f "${status_file}" ]] && grep -Eq '^[[:space:]]*status: completed$' "${status_file}"
}

assert_completed_batch() {
    local first_task=$1
    local last_task=$2
    local task_index
    for ((task_index = first_task; task_index <= last_task; task_index += 1)); do
        if ! task_completed "${task_index}"; then
            printf -v status_file '%s/plan/status/task_%06d.yaml' "${RUN_DIRECTORY}" "${task_index}"
            echo "Task ${task_index} did not complete successfully: ${status_file}" >&2
            exit 1
        fi
    done
}

# Resume from the first task without a completed manifest after a monitor restart.
while (( next_task < task_count )) && task_completed "${next_task}"; do
    next_task=$((next_task + 1))
done
printf '%s\n' "${next_task}" >"${NEXT_TASK_FILE}"

while (( next_task < task_count )); do
    # The QoS permits six submitted tasks; wait rather than competing with other user jobs.
    while squeue --noheader --user "${USER}" | grep -q .; do
        sleep 30
    done

    last_task=$((next_task + TASK_LIMIT - 1))
    if (( last_task >= task_count )); then
        last_task=$((task_count - 1))
    fi

    job_id=$(CAMPAIGN_FILE="${CAMPAIGN_FILE}" sbatch --parsable \
        --time="${TASK_WALLTIME}" \
        --array="${next_task}-${last_task}%${PARALLELISM}" \
        "${REPOSITORY_ROOT}/assets/scripts/entropy/05_task_array.sbatch")
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

job_id=$(CAMPAIGN_FILE="${CAMPAIGN_FILE}" sbatch --parsable \
    "${REPOSITORY_ROOT}/assets/scripts/entropy/06_finalize_campaign.sbatch")
job_id=${job_id%%;*}
if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse the finalizer job ID: ${job_id}" >&2
    exit 1
fi
printf '%s\n' "${job_id}" >"${FINALIZER_JOB_FILE}"
echo "Submitted finalizer ${job_id}."
wait_for_job_completion "${job_id}"
assert_job_completed_successfully "${job_id}"
echo "Campaign ${CAMPAIGN_ID} completed."
