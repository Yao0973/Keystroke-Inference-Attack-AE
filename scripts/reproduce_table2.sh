#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XDG_CACHE_HOME="${ROOT_DIR}/.artifact_cache"
export MPLCONFIGDIR="${ROOT_DIR}/.artifact_cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "[artifact] Reproducing Table 2 placeholder results..."
python3 -m keystroke_artifact.runner table2 --output-dir "${ROOT_DIR}/outputs/table2"
echo "[artifact] Table 2 reproduction finished successfully."
