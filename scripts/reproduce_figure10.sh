#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export XDG_CACHE_HOME="${ROOT_DIR}/.artifact_cache"
export MPLCONFIGDIR="${ROOT_DIR}/.artifact_cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "[artifact] Reproducing paper Figure 10 data..."
python3 -m keystroke_artifact.runner figure10 --output-dir "${ROOT_DIR}/outputs/figure10"
echo "[artifact] Regenerating Figure 10 PNG/PDF from CSV outputs..."
python3 scripts/plot_figure10.py \
  --length-csv "${ROOT_DIR}/outputs/figure10/tables/figure10_sequence_length_recovery.csv" \
  --topk-csv "${ROOT_DIR}/outputs/figure10/tables/figure10_topk_sensitivity.csv" \
  --output-dir "${ROOT_DIR}/outputs/figure10"
echo "[artifact] Figure 10 reproduction finished successfully."
