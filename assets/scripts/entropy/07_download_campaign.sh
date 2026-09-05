#!/usr/bin/env bash

# Download one completed Entropy campaign into the analogous local workspace.
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
    echo "Usage: $0 <workspace-dir> <campaign-id> [remote-host] [remote-repository]" >&2
    exit 2
fi

WORKSPACE_INPUT=${1%/}
CAMPAIGN_ID=$2
# REMARK: `entropy` is an SSH host alias, not a hostname. Define it in ~/.ssh/config.
REMOTE_HOST=${3:-entropy}
REMOTE_REPOSITORY=${4:-repositories/MSIAutoEncoderWrapper}

if [[ "${WORKSPACE_INPUT}" = /* ]]; then
    echo "workspace-dir must be relative to the repository so local and remote layouts match." >&2
    exit 2
fi
if [[ ! "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    echo "Campaign ID may contain only letters, numbers, underscores, and hyphens." >&2
    exit 2
fi

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOCAL_REPOSITORY=$(cd "${SCRIPT_DIRECTORY}/../../.." && pwd)
LOCAL_WORKSPACE=${LOCAL_REPOSITORY}/${WORKSPACE_INPUT}
LOCAL_RUN_DIRECTORY=${LOCAL_WORKSPACE}/configs/entropy-runs/${CAMPAIGN_ID}
REMOTE_WORKSPACE=${REMOTE_REPOSITORY%/}/${WORKSPACE_INPUT}

mkdir -p "${LOCAL_WORKSPACE}/models" "${LOCAL_RUN_DIRECTORY}"

# Model directories are campaign-scoped, so the include filters cannot fetch prior runs.
rsync -a --partial --append-verify --info=progress2 \
    --include='*/' \
    --include="*/${CAMPAIGN_ID}__task_*/***" \
    --exclude='*' \
    "${REMOTE_HOST}:${REMOTE_WORKSPACE}/models/" \
    "${LOCAL_WORKSPACE}/models/"

# Plan, task statuses, and task logs live inside the workspace and remain available offline.
rsync -a --partial --append-verify --info=progress2 \
    "${REMOTE_HOST}:${REMOTE_WORKSPACE}/configs/entropy-runs/${CAMPAIGN_ID}/" \
    "${LOCAL_RUN_DIRECTORY}/"
