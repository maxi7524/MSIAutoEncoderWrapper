#!/usr/bin/env bash
#SBATCH --job-name=msi-finalize
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --nodelist=asusgpu6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00

# Persist campaign artifacts and remove only this campaign's node-local staging directory.
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
if [[ ! -d "${SOURCE_WORKSPACE:-}" || ! -d "${LOCAL_WORKSPACE:-}" ]]; then
    echo "Campaign settings contain an unavailable source or local workspace." >&2
    exit 1
fi

# Persist only model directories produced by this campaign.
## LOCAL_WORKSPACE begins as a copy of SOURCE_WORKSPACE, so synchronizing its
## complete models/ or configs/ tree would overwrite files that predate this run.
if [[ -d "${LOCAL_WORKSPACE}/models" ]]; then
    while IFS= read -r -d '' model_directory; do
        context_name=$(basename "$(dirname "${model_directory}")")
        model_name=$(basename "${model_directory}")
        destination_directory=${SOURCE_WORKSPACE}/models/${context_name}/${model_name}
        if [[ -e "${destination_directory}" ]]; then
            echo "Refusing to overwrite existing model directory: ${destination_directory}" >&2
            exit 1
        fi
        mkdir -p "$(dirname "${destination_directory}")"
        rsync -a "${model_directory}/" "${destination_directory}/"
    done < <(
        find "${LOCAL_WORKSPACE}/models" -mindepth 2 -maxdepth 2 -type d \
            -name "${CAMPAIGN_ID}__*" -print0
    )
fi

# NODE_LOCAL_DIRECTORY is written by staging and is scoped to one validated campaign ID.
rm -rf -- "${NODE_LOCAL_DIRECTORY}"
echo "Finalized campaign ${CAMPAIGN_ID}; logs and task status are in ${RUN_DIRECTORY}."
