#!/usr/bin/env bash

# Stage and execute several campaigns serially on one Entropy node.
set -euo pipefail

if (( $# < 3 || $# % 2 == 0 )); then
    echo "Usage: $0 <workspace-dir> <campaign-id> <experiment-yaml> [<campaign-id> <experiment-yaml> ...]" >&2
    exit 2
fi

WORKSPACE_INPUT=$1
shift
SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY_ROOT=$(cd "${SCRIPT_DIRECTORY}/../../.." && pwd)
STAGE_SCRIPT=${SCRIPT_DIRECTORY}/02_stage_campaign.sbatch
ORCHESTRATOR_SCRIPT=${SCRIPT_DIRECTORY}/03_orchestrate_campaign.sh

wait_for_stage() {
    local job_id=$1
    local state=
    while squeue --noheader --jobs "${job_id}" 2>/dev/null | grep -q .; do
        sleep 30
    done
    for _ in $(seq 1 12); do
        state=$(sacct --noheader --allocations --jobs "${job_id}" --format=State | awk 'NR == 1 {print $1}')
        if [[ -n "${state}" ]]; then
            break
        fi
        sleep 5
    done
    if [[ "${state}" != COMPLETED ]]; then
        echo "Staging job ${job_id} ended with state '${state:-unknown}'." >&2
        exit 1
    fi
}

while (( $# > 0 )); do
    CAMPAIGN_ID=$1
    EXPERIMENT_YAML=$2
    shift 2
    if [[ "${WORKSPACE_INPUT}" = /* ]]; then
        WORKSPACE_DIRECTORY=${WORKSPACE_INPUT%/}
    else
        WORKSPACE_DIRECTORY=${REPOSITORY_ROOT}/${WORKSPACE_INPUT%/}
    fi
    RUN_DIRECTORY=${WORKSPACE_DIRECTORY}/configs/entropy-runs/${CAMPAIGN_ID}

    echo "Staging campaign ${CAMPAIGN_ID} using ${EXPERIMENT_YAML}."
    stage_job_id=$(cd "${REPOSITORY_ROOT}" && sbatch --parsable \
        "${STAGE_SCRIPT}" \
        "${CAMPAIGN_ID}" \
        "${EXPERIMENT_YAML}" \
        "${WORKSPACE_INPUT}" \
        "${RUN_DIRECTORY}")
    stage_job_id=${stage_job_id%%;*}
    if [[ ! "${stage_job_id}" =~ ^[0-9]+$ ]]; then
        echo "Could not parse the staging job ID: ${stage_job_id}" >&2
        exit 1
    fi
    wait_for_stage "${stage_job_id}"

    echo "Executing campaign ${CAMPAIGN_ID}."
    bash "${ORCHESTRATOR_SCRIPT}" "${RUN_DIRECTORY}"
done

echo "All requested campaigns completed."
