#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XDG_CACHE_HOME="${ROOT_DIR}/.artifact_cache"
export MPLCONFIGDIR="${ROOT_DIR}/.artifact_cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "[artifact] Reproducing paper Figure 9 confusion matrix..."
python3 -m keystroke_artifact.runner figure9 --output-dir "${ROOT_DIR}/outputs/figure9"
echo "[artifact] Figure 9 reproduction finished successfully."
