#!/usr/bin/env bash
# Refresh the minimum Rev2/L1.5 eval JSON set with matched-only MPJPE fields.
#
# This script does not train anything. It reruns only the four eval jobs needed
# for the manuscript tables/diagnostics after the L1.5 JSON schema patch:
#   G1  M9_RF2
#   G2  M0
#   G5a M9 no-flow
#   G5b M9 one-step flow
#
# Usage on the GPU server:
#   CKPT_M0=work_dirs/full_alation_20e/M0/latest.pth \
#   CKPT_M9=work_dirs/full_alation_20e/M9/latest.pth \
#   OUT_DIR=paper_assets/logs/rev2 \
#   bash scripts/gpu_run_rev2_l15_refresh.sh

set -euo pipefail

CKPT_M0=${CKPT_M0:?Set CKPT_M0 to M0 checkpoint path}
CKPT_M9=${CKPT_M9:?Set CKPT_M9 to M9 checkpoint path}
OUT_DIR=${OUT_DIR:-paper_assets/logs/rev2}
TOL=${TOL:-0.5}

M9_RF2_CFG=${M9_RF2_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py}
M0_CFG=${M0_CFG:-configs/wifi/petr_wifi.py}
M9_NOFLOW_CFG=${M9_NOFLOW_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_no_flow.py}
M9_RF1_CFG=${M9_RF1_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened.py}

mkdir -p "${OUT_DIR}"

echo "==== Verify static L1/L1.5 metric export markers ===="
python scripts/verify_l1_per_joint_mpjdle.py

run_eval() {
  local label="$1"
  local cfg="$2"
  local ckpt="$3"
  local json_out="$4"
  local log_out="$5"
  local expected="$6"

  echo
  echo "==== [${label}] ${cfg} ===="
  python tools/test.py \
    "${cfg}" \
    "${ckpt}" \
    --eval mpjpe \
    --metrics-out "${json_out}" \
    2>&1 | tee "${log_out}"

  python scripts/verify_l1_per_joint_mpjdle.py --eval-json "${json_out}"
  python - "${json_out}" "${expected}" "${TOL}" <<'PY'
import json
import sys

path, expected, tol = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
d = json.load(open(path))
mpjpe = float(d["mpjpe"])
drift = abs(mpjpe - expected)
print(f"{path}: mpjpe={mpjpe:.3f}, expected={expected:.3f}, drift={drift:.3f}")
if drift > tol:
    raise SystemExit(
        f"MPJPE sanity failed for {path}: drift {drift:.3f} > tolerance {tol:.3f}")
PY
}

run_eval "G1 M9_RF2" \
  "${M9_RF2_CFG}" "${CKPT_M9}" \
  "${OUT_DIR}/M9_RF2_eval.json" "${OUT_DIR}/G1_M9_RF2_eval.log" \
  "165.487"

run_eval "G2 M0" \
  "${M0_CFG}" "${CKPT_M0}" \
  "${OUT_DIR}/M0_eval.json" "${OUT_DIR}/G2_M0_eval.log" \
  "172.554"

run_eval "G5a M9 no-flow" \
  "${M9_NOFLOW_CFG}" "${CKPT_M9}" \
  "${OUT_DIR}/M9_no_flow_eval.json" "${OUT_DIR}/G5a_M9_no_flow_eval.log" \
  "167.736"

run_eval "G5b M9 RF1" \
  "${M9_RF1_CFG}" "${CKPT_M9}" \
  "${OUT_DIR}/M9_RF1_eval.json" "${OUT_DIR}/G5b_M9_RF1_eval.log" \
  "168.088"

echo
echo "==== Rev2/L1.5 refresh summary ===="
python - "${OUT_DIR}" <<'PY'
import json
import os
import sys

out_dir = sys.argv[1]
names = [
    "M9_RF2_eval.json",
    "M0_eval.json",
    "M9_no_flow_eval.json",
    "M9_RF1_eval.json",
]
print(f"{'file':24s} {'mpjpe':>8s} {'matched':>9s} {'missed':>7s}")
for name in names:
    path = os.path.join(out_dir, name)
    d = json.load(open(path))
    missed = d.get("missed_persons", d.get("missed", "n/a"))
    print(f"{name:24s} {d['mpjpe']:8.3f} {d['matched_mpjpe']:9.3f} {missed!s:>7s}")
PY

echo
echo "Outputs written to ${OUT_DIR}"
