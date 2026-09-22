#!/usr/bin/env bash

# PARAMS 
# These values must not exceed the Entropy QoS submission and GPU limits.
# REMARK
TASK_LIMIT=12
PARALLELISM=6
TASK_WALLTIME=${TASK_WALLTIME:-24:00:00}

# Submit bounded Slurm batches sequentially and finalize one staged campaign.
set -euo pipefail

restart=false
EXECUTION_NODE=
RUN_DIRECTORY_INPUT=
while (( $# > 0 )); do
    case $1 in
        --restart)
            restart=true
            ;;
        --nodelist)
            if (( $# < 2 )); then
                echo "Missing value for --nodelist." >&2
                exit 2
            fi
            EXECUTION_NODE=$2
            shift
            ;;
        --nodelist=*)
            EXECUTION_NODE=${1#*=}
            if [[ -z "${EXECUTION_NODE}" ]]; then
                echo "The --nodelist value must not be empty." >&2
                exit 2
            fi
            ;;
        --)
            shift
            if (( $# != 1 )); then
                echo "Usage: $0 [--restart] [--nodelist NODE] <run-directory>" >&2
                exit 2
            fi
            RUN_DIRECTORY_INPUT=$1
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [[ -n "${RUN_DIRECTORY_INPUT}" ]]; then
                echo "Usage: $0 [--restart] [--nodelist NODE] <run-directory>" >&2
                exit 2
            fi
            RUN_DIRECTORY_INPUT=$1
            ;;
    esac
    shift
done

if [[ -z "${RUN_DIRECTORY_INPUT}" ]]; then
    echo "Usage: $0 [--restart] [--nodelist NODE] <run-directory>" >&2
    exit 2
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
if [[ -n "${STAGING_NODE:-}" ]]; then
    if [[ -n "${EXECUTION_NODE}" && "${EXECUTION_NODE}" != "${STAGING_NODE}" ]]; then
        echo "Execution node must match the staging node: ${STAGING_NODE}." >&2
        exit 1
    fi
    EXECUTION_NODE=${STAGING_NODE}
fi

TASK_COUNT_FILE=${RUN_DIRECTORY}/task-count
TASK_JOB_HISTORY=${RUN_DIRECTORY}/task-array-job-ids
FINALIZER_JOB_FILE=${RUN_DIRECTORY}/finalizer-job-id
PYTHON=${REPOSITORY_ROOT}/.venv/bin/python
BATCH_PLANNER=${REPOSITORY_ROOT}/assets/scripts/entropy/campaign_task_batches.py

# Entropy run 

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
    rm -f -- "${TASK_JOB_HISTORY}" "${FINALIZER_JOB_FILE}"
fi

if [[ ! -x "${PYTHON}" || ! -f "${BATCH_PLANNER}" ]]; then
    echo "Missing campaign Python environment or batch planner." >&2
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

# Recompute the ready set after every array; completed parents unlock children.
while true; do
    batch=$("${PYTHON}" "${BATCH_PLANNER}" next \
        --plan-directory "${RUN_DIRECTORY}/plan" --limit "${TASK_LIMIT}" \
        --local-workspace "${LOCAL_WORKSPACE}")
    if [[ -z "${batch}" ]]; then
        break
    fi
    # The QoS permits six submitted tasks; wait rather than competing with other user jobs.
    while squeue --noheader --user "${USER}" | grep -q .; do
        sleep 30
    done

    task_submit_options=(
        --time="${TASK_WALLTIME}"
        --array="${batch}%${PARALLELISM}"
    )
    if [[ -n "${EXECUTION_NODE}" ]]; then
        task_submit_options+=(--nodelist="${EXECUTION_NODE}")
    fi

    job_id=$(CAMPAIGN_FILE="${CAMPAIGN_FILE}" sbatch --parsable \
        "${task_submit_options[@]}" \
        "${REPOSITORY_ROOT}/assets/scripts/entropy/03_1_task_array.sbatch")
    job_id=${job_id%%;*}
    if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
        echo "Could not parse the task-array job ID: ${job_id}" >&2
        exit 1
    fi

    printf '%s %s\n' "${job_id}" "${batch}" >>"${TASK_JOB_HISTORY}"
    echo "Submitted array ${job_id}: tasks ${batch}."
    wait_for_job_completion "${job_id}"
    "${PYTHON}" "${BATCH_PLANNER}" verify \
        --plan-directory "${RUN_DIRECTORY}/plan" --indices "${batch}" \
        --local-workspace "${LOCAL_WORKSPACE}"
done

finalizer_submit_options=()
if [[ -n "${EXECUTION_NODE}" ]]; then
    finalizer_submit_options+=(--nodelist="${EXECUTION_NODE}")
fi

job_id=$(CAMPAIGN_FILE="${CAMPAIGN_FILE}" sbatch --parsable \
    "${finalizer_submit_options[@]}" \
    "${REPOSITORY_ROOT}/assets/scripts/entropy/03_2_finalize_campaign.sbatch")
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
