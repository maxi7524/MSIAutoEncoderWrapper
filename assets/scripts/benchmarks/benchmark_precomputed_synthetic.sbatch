#!/usr/bin/env bash
#SBATCH --job-name=msi-precompute-benchmark
#SBATCH --partition=common
#SBATCH --qos=ms488923_common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Benchmark the persistent artifact on the same node-local staging used by tasks.
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: sbatch [--nodelist=<node>] --wait $0 <run-directory> <phase-name> [batches]" >&2
    exit 2
fi

RUN_DIRECTORY=$1
PHASE_NAME=$2
BATCHES=${3:-32}
REPOSITORY_ROOT=${REPOSITORY_ROOT:-${SLURM_SUBMIT_DIR:-}}

if [[ -z "${REPOSITORY_ROOT}" || ! -f "${REPOSITORY_ROOT}/pyproject.toml" ]]; then
    echo "Set REPOSITORY_ROOT to the repository containing pyproject.toml." >&2
    exit 2
fi
REPOSITORY_ROOT=$(cd "${REPOSITORY_ROOT}" && pwd)
PYTHON=${REPOSITORY_ROOT}/.venv/bin/python
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing ${PYTHON}. Run assets/scripts/entropy/01_setup_environment.sh first." >&2
    exit 1
fi

exec "${PYTHON}" \
    "${REPOSITORY_ROOT}/assets/scripts/benchmarks/benchmark_precomputed_synthetic.py" \
    --run-directory "${RUN_DIRECTORY}" \
    --phase-name "${PHASE_NAME}" \
    --batches "${BATCHES}"
