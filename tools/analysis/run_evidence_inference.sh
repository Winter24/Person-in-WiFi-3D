#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the rented GPU server.
# The script exports aggregate metrics and per-sample JSONL for inference-only
# paired bootstrap/decomposition analysis. It does not retrain any model.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${1:-paper_assets/logs/evidence_inference}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTPUT_DIR}"

run_eval() {
  local model_id="$1"
  local config="$2"
  local checkpoint="$3"
  local metrics_out="${OUTPUT_DIR}/${model_id}_eval.json"
  local per_sample_out="${OUTPUT_DIR}/${model_id}_per_sample.jsonl"

  echo "${model_id}: config=${config} checkpoint=${checkpoint}"
  "${PYTHON_BIN}" tools/test.py \
    "${PROJECT_ROOT}/${config}" \
    "${PROJECT_ROOT}/${checkpoint}" \
    --eval keypoints \
    --metrics-out "${metrics_out}" \
    --eval-options "per_sample_metrics_out=${per_sample_out}"
}

run_eval \
  "M0" \
  "work_dirs/full_alation_20e/M0/petr_wifi.py" \
  "work_dirs/full_alation_20e/M0/latest.pth"

run_eval \
  "M6" \
  "work_dirs/full_alation_20e/M6/wi_tidir_wifi_draft_mamba2_csi.py" \
  "work_dirs/full_alation_20e/M6/latest.pth"

run_eval \
  "M9_no_flow" \
  "configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_no_flow.py" \
  "work_dirs/full_alation_20e/M9/latest.pth"

run_eval \
  "M9_RF1" \
  "work_dirs/full_alation_20e/M9/wi_tidir_wifi_mamba2_flattened.py" \
  "work_dirs/full_alation_20e/M9/latest.pth"

run_eval \
  "M9_RF2" \
  "configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py" \
  "work_dirs/full_alation_20e/M9/latest.pth"

"${PYTHON_BIN}" tools/analysis/analyze_per_sample_metrics.py \
  --baseline "${OUTPUT_DIR}/M0_per_sample.jsonl" \
  --final "${OUTPUT_DIR}/M9_RF2_per_sample.jsonl" \
  --out "${OUTPUT_DIR}/M9_RF2_vs_M0_paired_bootstrap.json"

"${PYTHON_BIN}" tools/analysis/analyze_per_sample_metrics.py \
  --baseline "${OUTPUT_DIR}/M9_RF1_per_sample.jsonl" \
  --final "${OUTPUT_DIR}/M9_RF2_per_sample.jsonl" \
  --out "${OUTPUT_DIR}/M9_RF2_vs_M9_RF1_paired_bootstrap.json"

echo "Evidence inference outputs written to ${OUTPUT_DIR}"
