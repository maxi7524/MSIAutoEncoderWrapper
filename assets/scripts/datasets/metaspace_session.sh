#!/usr/bin/env bash

# This script must be sourced so the API key remains available to later commands
# in the current shell without being written to disk.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: source this script instead of executing it:" >&2
    echo "  source assets/scripts/initialization/metaspace_session.sh" >&2
    exit 1
fi

read -r -s -p "METASPACE API key: " METASPACE_API_KEY
printf '\n'

if [[ -z "${METASPACE_API_KEY}" ]]; then
    echo "Error: no API key was provided." >&2
    unset METASPACE_API_KEY
    return 1
fi

export METASPACE_API_KEY

if python -m msi_dataset_manager.sources.strategies.metaspace_authentication; then
    echo "METASPACE commands can now use this session."
    echo "Run 'unset METASPACE_API_KEY' to end the session."
else
    echo "The API key was removed from this shell because validation failed." >&2
    unset METASPACE_API_KEY
    return 1
fi
