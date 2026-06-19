#!/usr/bin/env bash
# Tier 2 GPU run — G3 (per-receiver ablation) + G5 (flow-step ablation refresh).
# See docs/superpowers/specs/2026-06-19-gpu-runbook-rev2.md.
#
# Pre-req: gpu_run_tier1.sh must have passed (sanity confirms checkpoint/data integrity).
#
# Usage:
#   CKPT_M9=/workspace/ckpt/M9/latest.pth bash scripts/gpu_run_tier2.sh
#
# Expected wall time: ~20 minutes on RTX 3090.

set -euo pipefail

CKPT_M9=${CKPT_M9:?Set CKPT_M9 to M9 checkpoint path}
OUT_DIR=${OUT_DIR:-paper_assets/logs/rev2}
M9_RF2_CFG=${M9_RF2_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_rf_2step.py}
M9_NOFLOW_CFG=${M9_NOFLOW_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened_eval_no_flow.py}
M9_RF1_CFG=${M9_RF1_CFG:-configs/wifi/wi_tidir_wifi_mamba2_flattened.py}

mkdir -p "${OUT_DIR}"

# ----------- Receiver-masking sanity -----------
echo "==== [G3-sanity] Confirm receiver-mask hook reproduces full-receiver MPJPE ===="
python tools/eval_receiver_subset.py \
  "${M9_RF2_CFG}" "${CKPT_M9}" \
  --rx-keep 0 1 2 \
  --out "${OUT_DIR}/M9_RF2_R1R2R3_sanity.json"

SANITY_MPJPE=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["mpjpe"])' \
              "${OUT_DIR}/M9_RF2_R1R2R3_sanity.json")
echo "Sanity MPJPE = ${SANITY_MPJPE} (expect ~165.487, tolerance 1.0)"
python -c "import sys; m=float('${SANITY_MPJPE}'); assert abs(m-165.487)<1.0, \
           f'Receiver-mask sanity FAILED: {m} drifts >1mm from 165.487. STOP.'"

# ----------- G3: receiver subsets -----------
declare -A SUBSETS=(
  [R1R2]="0 1"
  [R1R3]="0 2"
  [R2R3]="1 2"
  [R2]="1"
)
for NAME in "${!SUBSETS[@]}"; do
  IDXS=${SUBSETS[$NAME]}
  echo
  echo "==== [G3-${NAME}] Eval M9_RF2 with Rx={${IDXS}} ===="
  python tools/eval_receiver_subset.py \
    "${M9_RF2_CFG}" "${CKPT_M9}" \
    --rx-keep ${IDXS} \
    --out "${OUT_DIR}/M9_RF2_${NAME}_eval.json"
done

# ----------- G5: flow-step refresh (per-joint added) -----------
echo
echo "==== [G5a] Eval M9 with flow disabled ===="
python tools/test.py \
  "${M9_NOFLOW_CFG}" \
  "${CKPT_M9}" --eval mpjpe \
  --metrics-out "${OUT_DIR}/M9_no_flow_eval.json"

echo
echo "==== [G5b] Eval M9 with 1-step rectified flow (existing M9 row, refreshed) ===="
python tools/test.py \
  "${M9_RF1_CFG}" \
  "${CKPT_M9}" --eval mpjpe \
  --metrics-out "${OUT_DIR}/M9_RF1_eval.json"

echo
echo "==== Tier 2 summary ===="
python - "${OUT_DIR}" <<'PY'
import glob, json, os, sys
out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, '*.json')))
print(f"{'file':60s}  {'mpjpe':>8s}  {'1P':>7s}  {'2P':>7s}  {'3P':>7s}")
for f in files:
    d = json.load(open(f))
    print(f"{f:60s}  {d['mpjpe']:8.3f}  "
          f"{d.get('mpjpe_1p', 0):7.3f}  "
          f"{d.get('mpjpe_2p', 0):7.3f}  "
          f"{d.get('mpjpe_3p', 0):7.3f}")
PY

ls -la "${OUT_DIR}/"
