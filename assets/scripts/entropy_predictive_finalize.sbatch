#!/usr/bin/env bash
#SBATCH --job-name=msi-predictive-finalize
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --nodelist=asusgpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00

# Persist generated artifacts and remove only this campaign's RAM directory.
set -euo pipefail

if [[ -z "${CAMPAIGN_ID:-}" ]] || [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "CAMPAIGN_ID must be a valid staging-job identifier." >&2
    exit 2
fi

REPOSITORY_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SOURCE_WORKSPACE=${REPOSITORY_ROOT}/data/kidney_workspace
RAM_DIRECTORY=/dev/shm/${USER}/msi-wrapper/${CAMPAIGN_ID}
LOCAL_WORKSPACE=${RAM_DIRECTORY}/kidney_workspace
RUN_DIRECTORY=${HOME}/entropy-runs/kidney-architecture-predictive/${CAMPAIGN_ID}

if [[ ! -d "${LOCAL_WORKSPACE}" ]]; then
    echo "Staged workspace is unavailable: ${LOCAL_WORKSPACE}" >&2
    exit 1
fi

mkdir -p "${SOURCE_WORKSPACE}/models" "${SOURCE_WORKSPACE}/configs"
if [[ -d "${LOCAL_WORKSPACE}/models" ]]; then
    rsync -a "${LOCAL_WORKSPACE}/models/" "${SOURCE_WORKSPACE}/models/"
fi
if [[ -d "${LOCAL_WORKSPACE}/configs" ]]; then
    rsync -a "${LOCAL_WORKSPACE}/configs/" "${SOURCE_WORKSPACE}/configs/"
fi

# The path is constructed from a validated user identifier, never from input paths.
rm -rf -- "${RAM_DIRECTORY}"
echo "Finalized campaign ${CAMPAIGN_ID}; logs and task status are in ${RUN_DIRECTORY}."
